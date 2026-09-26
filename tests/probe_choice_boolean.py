#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two independent noul questions for a large-choice failure."""

import argparse
import os

from test_jev_api_smoke import request_api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("LOCAL_API_KEY"))
    parser.add_argument("--number", type=int, default=65)
    parser.add_argument("--other", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--choice", action="store_true", help="Also test the full choice request")
    parser.add_argument("--batch", action="store_true", help="Also test independent noul questions in one request")
    args = parser.parse_args()

    for candidate in (args.number, args.other):
        payload = {
            "model": "jev-latest",
            "state": f"The number is {args.number}.",
            "questions": {"match": {
                "type": "noul",
                "instructions": f"Is item_{candidate} (Item {candidate}) the selected item?",
            }},
        }
        response, status, elapsed_ms = request_api(
            args.url, args.token, payload, args.timeout,
        )
        if status != 200:
            raise SystemExit(f"item_{candidate}: HTTP {status}: {response}")
        probability = response["answers"]["match"]["noul"]
        print(f"item_{candidate}: noul={probability:.6f} | "
              f"predicted={probability >= 0.5} | {elapsed_ms:.1f} ms")

    if args.choice:
        payload = {
            "model": "jev-latest",
            "state": f"The number is {args.number}.",
            "questions": {"item": {
                "type": "choice",
                "instructions": f"The number is {args.number}. Choose the correspondent item with the selected number.",
                "criteria": {f"item_{i}": f"Item {i}" for i in range(1, args.number + 1)},
            }},
        }
        response, status, elapsed_ms = request_api(
            args.url, args.token, payload, args.timeout,
        )
        if status != 200:
            raise SystemExit(f"choice: HTTP {status}: {response}")
        answer = response["answers"]["item"]
        top = sorted(answer["probabilities"].items(), key=lambda item: item[1], reverse=True)[:5]
        print(f"choice: selected={answer['choice']} | expected=item_{args.number} "
              f"| top={top} | {elapsed_ms:.1f} ms")

    if args.batch:
        candidates = [args.number, args.other]
        payload = {
            "model": "jev-latest",
            "state": f"The number is {args.number}.",
            "questions": {
                f"item_{candidate}": {
                    "type": "noul",
                    "instructions": f"Is item_{candidate} (Item {candidate}) the selected item?",
                }
                for candidate in candidates
            },
        }
        response, status, elapsed_ms = request_api(args.url, args.token, payload, args.timeout)
        if status != 200:
            raise SystemExit(f"batch: HTTP {status}: {response}")
        print(f"batch: {[(key, answer['noul']) for key, answer in response['answers'].items()]} "
              f"| {elapsed_ms:.1f} ms")


if __name__ == "__main__":
    main()
