#!/usr/bin/env python3
"""
Retrain CatBoost from an explicit SQLite copy using the cleaned feature schema.
"""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import database
import tuner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", required=True, help="Path to the SQLite copy to train from")
    parser.add_argument("--model-path", required=True, help="Where to save the cleaned model")
    parser.add_argument(
        "--train-dir",
        default=None,
        help="Optional CatBoost train_dir for temporary artifacts",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=None,
        help="Override deployment threshold for this run",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Do not append this run to training_log in the copied database",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    conn = database.init_db(args.db_path)
    result = tuner.train_model(
        conn,
        model_path=args.model_path,
        min_accuracy=args.min_accuracy,
        log_run=not args.no_log,
        train_dir=args.train_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
