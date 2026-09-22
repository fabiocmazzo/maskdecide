# SPDX-License-Identifier: Apache-2.0
"""Compare decoder accuracy and warm decoding time on a small diagnostic suite.

Run with the API server stopped to avoid loading a second model into GPU memory:
    uv run python scripts/compare_decoders.py --output /tmp/decoder-comparison.json

This is not a representative benchmark. Timings exclude prompt construction,
tokenization, model loading, and HTTP. Expected answers never enter the decoder.
"""

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from maskdecide import api
from maskdecide.decoding import decode
from test_jev_api_smoke import CASES, choice, noul, score


def additional_cases():
    """Fresh diagnostic examples, separate from the existing smoke suite."""
    cases = []
    for value in [1, 2, 5, 6, 9]:
        cases.append({
            "name": f"Additional numeric range: {value}",
            "state": {"reading": value},
            "questions": {"band": score("Select the inclusive interval containing reading.", [
                "From 0 through 2", "From 3 through 6", "From 7 through 9",
            ])},
            "expected": {"band": 0 if value <= 2 else 1 if value <= 6 else 2},
        })
    for latest in ["open", "closed"]:
        earlier = "closed" if latest == "open" else "open"
        cases.append({
            "name": f"Additional status transition: {earlier} to {latest}",
            "state": f"At 10:00 the gate was {earlier}. At 11:00 it changed to {latest}. It is now 11:05; there have been no further changes.",
            "questions": {
                "now_open": noul("Is the gate open now?"),
                "was_open": noul("Was the gate open at 10:00?"),
            },
            "expected": {"now_open": latest == "open", "was_open": earlier == "open"},
        })
    for count in [26, 27]:
        for target in [1, 13]:
            cases.append({
                "name": f"Additional choice: {count} options, target {target}",
                "state": f"The designated container number is {target}.",
                "questions": {"container": choice("Choose the designated container.", {
                    f"container_{i}": f"Container number {i}" for i in range(1, count + 1)
                })},
                "expected": {"container": f"container_{target}"},
            })
    return cases


async def compare(args):
    records = []
    async with api.lifespan(api.app):
        for suite, cases in [("smoke", CASES), ("additional", additional_cases())]:
            for case in cases:
                request = api.JevRequest(state=case["state"], questions=case["questions"])
                api.validate_jev_request(request)
                prompt, slots, plans = api.build_jev(request)
                inputs, positions = api.build_masked_input(prompt, slots)
                ids = api.letter_token_ids("".join(dict.fromkeys("".join(s.allowed for s in slots))))
                candidates = [[ids[letter] for letter in slot.allowed] for slot in slots]
                for mode in ["one_pass", "iterative"]:
                    # Warm this input shape and decoder before measuring.
                    decode(api.model, inputs, positions, candidates, mask_id=api.MASK_ID,
                           mode=mode, threshold=args.threshold)
                    for repeat in range(args.repeat):
                        torch.cuda.synchronize()
                        started = perf_counter()
                        result = decode(api.model, inputs, positions, candidates, mask_id=api.MASK_ID,
                                        mode=mode, threshold=args.threshold)
                        torch.cuda.synchronize()
                        elapsed = (perf_counter() - started) * 1000
                        labels = [slot.allowed[i] for slot, i in zip(slots, result.winners, strict=True)]
                        answers = api.create_jev_answers(plans, labels, result.logits, result.probabilities)
                        predicted = {qid: (answer["noul"] >= 0.5 if answer["type"] == "noul"
                                           else answer[answer["type"]]) for qid, answer in answers.items()}
                        records.append({
                            "suite": suite, "case": case["name"], "mode": mode, "repeat": repeat + 1,
                            "predicted": predicted, "expected": case["expected"],
                            "correct": sum(predicted[k] == v for k, v in case["expected"].items()),
                            "total": len(predicted), "decode_ms": elapsed,
                            "forward_passes": result.forward_passes, "input_tokens": inputs.shape[1],
                        })
                print(f"Checked {suite}: {case['name']}", flush=True)

        summaries = []
        for suite in ["smoke", "additional"]:
            for mode in ["one_pass", "iterative"]:
                selected = [r for r in records if r["suite"] == suite and r["mode"] == mode]
                latencies = sorted(r["decode_ms"] for r in selected)
                summary = {
                    "suite": suite, "mode": mode, "repeat": args.repeat,
                    "correct": sum(r["correct"] for r in selected),
                    "total": sum(r["total"] for r in selected),
                    "median_decode_ms": statistics.median(latencies),
                    "mean_decode_ms": statistics.mean(latencies),
                    "max_forward_passes": max(r["forward_passes"] for r in selected),
                }
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
        report = {
            "model": api.MODEL_ID,
            "model_revision": getattr(api.model.config, "_commit_hash", None),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
            "threshold": args.threshold, "summaries": summaries, "records": records,
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Report written to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--output", type=Path, default=Path("/tmp/maskdecide-decoder-comparison.json"))
    args = parser.parse_args()
    if args.repeat < 1 or not 0 < args.threshold <= 1:
        parser.error("--repeat must be >= 1 and --threshold must be in (0, 1]")
    asyncio.run(compare(args))


if __name__ == "__main__":
    main()
