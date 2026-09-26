#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare two candidate items in both orders using the ordinary choice path."""

import argparse
import math
import os

from test_jev_api_smoke import request_api


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("LOCAL_API_KEY"))
    parser.add_argument("--target", type=int, default=65)
    parser.add_argument("--other", type=int, default=50)
    parser.add_argument("--count", type=int, default=0, help="Run a complete pairwise tournament")
    args = parser.parse_args()

    def compare(left: int, right: int) -> tuple[int, float]:
        probabilities = []
        for order in ((left, right), (right, left)):
            criteria = {f"item_{i}": f"Item {i}" for i in order}
            response, status, _ = request_api(args.url, args.token, {
                "model": "jev-latest", "state": f"The number is {args.target}.",
                "questions": {"item": {
                    "type": "choice", "instructions": (
                        f"The number is {args.target}. Choose the correspondent item "
                        "with the selected number."
                    ),
                    "criteria": criteria,
                }},
            }, 30)
            if status != 200:
                raise SystemExit(f"HTTP {status}: {response}")
            probabilities.append(response["answers"]["item"]["probabilities"][f"item_{left}"])
        margin = sum(math.log(p / (1 - p)) for p in probabilities) / 2
        return (left if margin > 0 else right), margin

    if args.count:
        competitors = list(range(1, args.count + 1))
        rounds = 0
        while len(competitors) > 1:
            next_round = []
            for index in range(0, len(competitors) - 1, 2):
                winner, margin = compare(competitors[index], competitors[index + 1])
                next_round.append(winner)
                if args.target in competitors[index:index + 2]:
                    print(f"target_pair={competitors[index:index+2]} winner={winner} margin={margin:+.4f}")
            if len(competitors) % 2:
                next_round.append(competitors[-1])
            competitors = next_round
            rounds += 1
        print(f"tournament count={args.count} target={args.target} winner={competitors[0]} rounds={rounds}")
        return

    for order in ((args.target, args.other), (args.other, args.target)):
        criteria = {f"item_{i}": f"Item {i}" for i in order}
        response, status, elapsed = request_api(args.url, args.token, {
            "model": "jev-latest",
            "state": f"The number is {args.target}.",
            "questions": {"item": {
                "type": "choice",
                "instructions": (
                    f"The number is {args.target}. Choose the correspondent item "
                    "with the selected number."
                ),
                "criteria": criteria,
            }},
        }, 30)
        if status != 200:
            raise SystemExit(f"HTTP {status}: {response}")
        answer = response["answers"]["item"]
        print(f"order={order} selected={answer['choice']} "
              f"target_probability={answer['probabilities'][f'item_{args.target}']:.4f} "
              f"{elapsed:.1f} ms")


if __name__ == "__main__":
    main()
