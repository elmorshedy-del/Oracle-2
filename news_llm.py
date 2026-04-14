"""
Signal 2: News + LLM Event Detection

1. Polls NewsAPI for relevant headlines (every N minutes)
2. Filters for new unseen headlines matching keywords
3. Only calls Claude when a genuinely new headline appears
4. Claude assesses: probability BTC goes up/down, confidence, reasoning
5. Compares LLM probability to current Polymarket price → edge calculation

Cost control:
- Only calls Claude on NEW headlines (not on a timer)
- ~30-60 calls/day typical = $4-8/month on Sonnet
- Skips duplicate / stale headlines
"""

import asyncio
import time
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone

import aiohttp

import config

log = logging.getLogger("news_llm")


class NewsLLMSignal:
    """
    Combines NewsAPI headline ingestion with Claude probability assessment.
    Only fires when genuinely new, relevant news appears.
    """

    def __init__(self):
        self._session = None
        self._seen_headlines = set()  # hash of title to avoid duplicates
        self._last_poll = 0
        self._last_assessment = None  # most recent LLM output
        self._assessment_age = 0
        self._call_count = 0  # track API spend

        # State
        self.active = False
        self.edge = 0.0           # LLM prob - market prob
        self.llm_probability = None  # LLM's assessed probability
        self.confidence = 0.0
        self.direction = 0        # +1 bullish, -1 bearish, 0 neutral
        self.headline = ""
        self.reasoning = ""

    async def start(self):
        self._session = aiohttp.ClientSession()
        if not config.NEWSAPI_KEY:
            log.warning("No NEWSAPI_KEY set — news signal disabled")
            return
        if not config.ANTHROPIC_API_KEY:
            log.warning("No ANTHROPIC_API_KEY set — LLM signal disabled")
            return
        self.active = True
        log.info("News+LLM signal started")

    async def stop(self):
        if self._session:
            await self._session.close()

    async def update(self, poly_mid: float = None):
        """
        Check for new headlines. If found, call Claude.
        poly_mid: current Polymarket YES mid price (0-1)
        """
        if not self.active:
            return

        now = time.time()
        if now - self._last_poll < config.NEWS_POLL_INTERVAL:
            return

        self._last_poll = now

        # Step 1: Fetch headlines
        new_headlines = await self._fetch_news()

        if not new_headlines:
            # Decay the signal over time (stale assessments lose value)
            if self._last_assessment and now - self._assessment_age > 300:
                self.edge = 0.0
                self.confidence = 0.0
                self.direction = 0
            return

        # Step 2: Pick the most impactful headline
        # Simple heuristic: newest first, prefer headlines with strong keywords
        best = self._rank_headlines(new_headlines)

        if not best:
            return

        # Step 3: Call Claude
        assessment = await self._assess_with_llm(best, poly_mid)

        if assessment:
            self._last_assessment = assessment
            self._assessment_age = now
            self._apply_assessment(assessment, poly_mid)

    async def _fetch_news(self) -> list:
        """Fetch recent headlines from NewsAPI, filter for relevance."""
        try:
            # Build query from keywords
            query = " OR ".join(config.NEWS_KEYWORDS[:5])  # API limits query length
            from_date = (
                datetime.now(timezone.utc) - timedelta(minutes=config.NEWS_MAX_AGE_MINUTES)
            ).strftime("%Y-%m-%dT%H:%M:%S")

            params = {
                "q": query,
                "from": from_date,
                "sortBy": "publishedAt",
                "language": "en",
                "pageSize": 10,
                "apiKey": config.NEWSAPI_KEY,
            }

            async with self._session.get(
                config.NEWSAPI_URL, params=params,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    articles = data.get("articles", [])

                    # Filter: only new headlines we haven't seen
                    new_ones = []
                    for article in articles:
                        title = article.get("title", "")
                        desc = article.get("description", "")
                        source = article.get("source", {}).get("name", "")
                        published = article.get("publishedAt", "")

                        # Hash to detect duplicates
                        h = hashlib.md5(title.encode()).hexdigest()
                        if h in self._seen_headlines:
                            continue
                        self._seen_headlines.add(h)

                        # Check relevance: at least one keyword in title or desc
                        text = (title + " " + desc).lower()
                        if any(kw in text for kw in config.NEWS_KEYWORDS):
                            new_ones.append({
                                "title": title,
                                "description": desc,
                                "source": source,
                                "published": published,
                            })

                    if new_ones:
                        log.info(f"  Found {len(new_ones)} new headline(s)")
                    return new_ones

                elif resp.status == 429:
                    log.warning("NewsAPI rate limited — backing off")
                    return []
                else:
                    log.debug(f"NewsAPI returned {resp.status}")
                    return []

        except Exception as e:
            log.debug(f"NewsAPI fetch error: {e}")
            return []

    def _rank_headlines(self, headlines: list) -> dict:
        """Pick the most impactful headline. Simple keyword scoring."""
        high_impact_keywords = [
            "crash", "surge", "plunge", "soar", "halt", "ban",
            "approve", "reject", "hack", "breach", "default",
            "emergency", "war", "sanction", "tariff",
            "fed", "rate cut", "rate hike", "inflation",
        ]

        best = None
        best_score = 0

        for h in headlines:
            text = (h["title"] + " " + h.get("description", "")).lower()
            score = sum(1 for kw in high_impact_keywords if kw in text)
            # Boost for reputable sources
            source = h.get("source", "").lower()
            if any(s in source for s in ["reuters", "bloomberg", "ap", "wsj", "cnbc"]):
                score += 2
            if score > best_score:
                best_score = score
                best = h

        # If no headline scored above 0, take the newest one
        if best is None and headlines:
            best = headlines[0]

        return best

    async def _assess_with_llm(self, headline: dict, poly_mid: float) -> dict:
        """
        Call Claude to assess the headline's impact on BTC probability.

        Returns: {probability, confidence, direction, reasoning}
        """
        title = headline.get("title", "")
        desc = headline.get("description", "")
        source = headline.get("source", "")
        published = headline.get("published", "")

        market_context = ""
        if poly_mid is not None:
            market_context = (
                f"\nCurrent Polymarket BTC up/down market: YES (up) is trading "
                f"at {poly_mid:.2f} ({poly_mid*100:.0f}% implied probability)."
            )

        prompt = f"""You are a quantitative prediction market analyst. Assess this headline's 
impact on short-term Bitcoin price direction (next 5-60 minutes).

HEADLINE: {title}
DESCRIPTION: {desc}
SOURCE: {source}
PUBLISHED: {published}
{market_context}

Respond ONLY with valid JSON, no other text:
{{
    "probability_btc_up": <float 0.0 to 1.0>,
    "confidence": <float 0.0 to 1.0, how confident you are in your assessment>,
    "impact_magnitude": <"none", "low", "medium", "high">,
    "reasoning": "<one sentence>"
}}

Rules:
- 0.5 = no impact / neutral
- >0.5 = bullish for BTC
- <0.5 = bearish for BTC
- confidence reflects how certain you are, not how big the move will be
- If the headline is irrelevant to BTC, return probability=0.5, confidence=0.1"""

        try:
            headers = {
                "Content-Type": "application/json",
                "x-api-key": config.ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            }
            payload = {
                "model": config.LLM_MODEL,
                "max_tokens": config.LLM_MAX_TOKENS,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
            }

            async with self._session.post(
                "https://api.anthropic.com/v1/messages",
                json=payload, headers=headers,
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                self._call_count += 1

                if resp.status == 200:
                    data = await resp.json()
                    text = data.get("content", [{}])[0].get("text", "")

                    # Parse JSON from response
                    text = text.strip()
                    if text.startswith("```"):
                        text = text.split("```")[1]
                        if text.startswith("json"):
                            text = text[4:]

                    assessment = json.loads(text)

                    log.info(
                        f"  LLM assessment #{self._call_count}: "
                        f"P(up)={assessment.get('probability_btc_up', '?')}, "
                        f"conf={assessment.get('confidence', '?')}, "
                        f"impact={assessment.get('impact_magnitude', '?')}"
                    )
                    log.info(f"  └─ Headline: {title[:80]}")
                    log.info(f"  └─ Reasoning: {assessment.get('reasoning', '?')}")

                    return assessment

                elif resp.status == 429:
                    log.warning("Claude API rate limited")
                    return None
                else:
                    body = await resp.text()
                    log.warning(f"Claude API returned {resp.status}: {body[:200]}")
                    return None

        except json.JSONDecodeError as e:
            log.warning(f"Failed to parse LLM JSON: {e}")
            return None
        except Exception as e:
            log.error(f"LLM assessment error: {e}")
            return None

    def _apply_assessment(self, assessment: dict, poly_mid: float):
        """Convert LLM assessment into actionable signal."""
        prob = assessment.get("probability_btc_up", 0.5)
        conf = assessment.get("confidence", 0.0)
        impact = assessment.get("impact_magnitude", "none")

        # Direction
        if prob > 0.55:
            self.direction = 1
        elif prob < 0.45:
            self.direction = -1
        else:
            self.direction = 0

        self.llm_probability = prob
        self.confidence = conf
        self.reasoning = assessment.get("reasoning", "")

        # Edge vs market
        if poly_mid is not None:
            self.edge = prob - poly_mid  # positive = LLM thinks market is too low on YES
        else:
            self.edge = 0.0

        # Only signal if edge AND confidence exceed thresholds
        if abs(self.edge) < config.LLM_EDGE_THRESHOLD:
            self.edge = 0.0
            self.direction = 0
        if conf < config.LLM_CONFIDENCE_THRESHOLD:
            self.edge = 0.0
            self.direction = 0

        if self.direction != 0:
            log.info(
                f"  ★ NEWS SIGNAL: direction={'↑' if self.direction > 0 else '↓'}, "
                f"edge={self.edge:+.2%}, conf={conf:.2f}, "
                f"LLM prob={prob:.2f} vs market={poly_mid:.2f}"
            )

    def get_signal(self) -> dict:
        """Return current news/LLM signal for the regime classifier."""
        return {
            "news_active": self.active and self.direction != 0,
            "news_direction": self.direction,
            "news_edge": round(self.edge, 4),
            "news_confidence": round(self.confidence, 3),
            "news_llm_prob": self.llm_probability,
            "news_headline": self.headline,
            "news_reasoning": self.reasoning,
            "news_call_count": self._call_count,
        }
