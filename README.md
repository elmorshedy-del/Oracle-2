# Polymarket 4-Mode Paper Trading Bot

An informed market making system for Polymarket prediction markets. Combines real-time BTC price data (Binance), Polymarket orderbook analysis, Kalshi cross-platform comparison, and LLM-powered news assessment into a unified 4-mode trading engine with automatic CatBoost model tuning.

**All paper mode. No real money. No wallet needed.**

## Quick Start

```bash
git clone https://github.com/YOUR_USERNAME/polymarket-bot.git
cd polymarket-bot
pip install -r requirements.txt

# Optional: add API keys for news/LLM signal
cp .env.example .env

python main.py
# Dashboard → http://localhost:8080
```

## Deploy to Railway

1. Push to GitHub
2. Connect repo in Railway dashboard
3. Add env vars: `NEWSAPI_KEY`, `ANTHROPIC_API_KEY`
4. Deploy — auto-detects Procfile and railway.json

## Architecture

```
Binance WS ──┐     ┌────────────┐
Polymarket ──┤────►│  4-MODE    │────► Paper Trader
Kalshi ──────┤     │  REGIME    │────► Risk Manager
NewsAPI ─────┤     │ CLASSIFIER │────► SQLite Logger
Claude LLM ──┘     └─────┬──────┘────► Dashboard
                         │
                   ┌─────▼──────┐
                   │  CATBOOST  │
                   │  AUTO-TUNE │
                   └────────────┘
```

## Modes

| Mode | Trigger | Action | Freq |
|------|---------|--------|------|
| 1 Quiet | No signals | Symmetric quotes | ~80% |
| 2 Lean | BTC momentum > 0.10% | Asymmetric quotes | ~15% |
| 3 Event | LLM edge > 8% or BTC spike | Aggressive one-sided | ~4% |
| 4 Arb | Cross-platform spread > 3¢ | Both legs | Rare |

## Environment Variables

| Variable | Required | Note |
|----------|----------|------|
| `NEWSAPI_KEY` | Optional | Free tier works |
| `ANTHROPIC_API_KEY` | Optional | ~$4-8/month on Sonnet |
| `PORT` | Optional | Default 8080 |

## Disclaimer

Paper trading only. No real money at risk. Not financial advice.
