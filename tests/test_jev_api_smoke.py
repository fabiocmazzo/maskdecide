#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""HTTP smoke checks for the local Jev adapter: contract and decision accuracy.

Usage:
    python test_jev_api_smoke.py
    python test_jev_api_smoke.py --url http://127.0.0.1:8000 --repeat 3
    python test_jev_api_smoke.py --token local-test-key
    python test_jev_api_smoke.py --list
    python test_jev_api_smoke.py --case "Structured" --repeat 3

Uses only the standard library. The FastAPI server must already be running.
"""

import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
import random



def noul(instructions):
    return {"type": "noul", "instructions": instructions}


def choice(instructions, options):
    return {"type": "choice", "instructions": instructions, "criteria": options}


def score(instructions, levels):
    return {"type": "score", "instructions": instructions, "criteria": levels}


CASES = [
    {
        "name": "Boolean: true statement",
        "state": "2 + 2 = 4.",
        "questions": {"is_four": noul("Is the result of 2 + 2 equal to 4?")},
        "expected": {"is_four": True},
    },
    {
        "name": "Boolean: false statement",
        "state": "2 + 2 = 4.",
        "questions": {"is_five": noul("Is the result of 2 + 2 equal to 5?")},
        "expected": {"is_five": False},
    },
    {
        "name": "Negation: login issue resolved",
        "state": "The customer successfully logged in. Authentication now works.",
        "questions": {"login_pending": noul("Is the customer still having an authentication problem?")},
        "expected": {"login_pending": False},
    },
    {
        "name": "Negation: login issue unresolved",
        "state": "The customer still cannot log in. Authentication keeps failing.",
        "questions": {"login_pending": noul("Is the customer still having an authentication problem?")},
        "expected": {"login_pending": True},
    },
    {
        "name": "Two booleans: mention versus addressee",
        "state": "The person says: 'Hanna, have you seen Jeff?' Hanna and Jeff are both present.",
        "questions": {
            "hanna": noul("Is the person speaking directly to Hanna?"),
            "jeff": noul("Is the person speaking directly to Jeff rather than merely mentioning him?"),
        },
        "expected": {"hanna": True, "jeff": False},
    },
    {
        "name": "Choice: original order",
        "state": "The result of 2 + 2 is 4.",
        "questions": {
            "result": choice(
                "Which option gives the correct result of 2 + 2?",
                {"three": "3", "four": "4", "five": "5"},
            )
        },
        "expected": {"result": "four"},
    },
    {
        "name": "Choice: reversed order",
        "state": "The result of 2 + 2 is 4.",
        "questions": {
            "result": choice(
                "Which option gives the correct result of 2 + 2?",
                {"five": "5", "four": "4", "three": "3"},
            )
        },
        "expected": {"result": "four"},
    },
    {
        "name": "Score: check scale and distribution",
        "state": "Observed number: 5. Rules: 0 to 3 = low; 4 to 7 = medium; 8 to 10 = high.",
        "questions": {
            "range": {
                "type": "score",
                "instructions": "Which range contains the observed number?",
                "criteria": ["Low: 0 to 3", "Medium: 4 to 7", "High: 8 to 10"],
            }
        },
        "expected": {"range": 1},  # level index, NOT a continuous score
    },
]


# Expected decisions follow explicit facts or rules in each case. Difficult
# cases remain in the suite even when the current model gets them wrong.
CASES.extend([
    {
        "name": "Negation: denied action versus completed action",
        "state": "Maya did not cancel the subscription. She updated the billing address.",
        "questions": {
            "cancelled": noul("Did Maya cancel the subscription?"),
            "updated": noul("Did Maya update the billing address?"),
        },
        "expected": {"cancelled": False, "updated": True},
    },
    {
        "name": "Temporal: latest status overrides earlier status",
        "state": "09:00: The service was offline. 09:15: The service recovered. It is now 09:20 and remains online.",
        "questions": {
            "offline_now": noul("Is the service offline now?"),
            "offline_before": noul("Was the service offline at 09:00?"),
        },
        "expected": {"offline_now": False, "offline_before": True},
    },
    {
        "name": "Boolean rubric: explicit true and false criteria",
        "state": "The invoice is 12 days overdue.",
        "questions": {"escalate": {
            **noul("Should this invoice be escalated under the rubric?"),
            "criteria": {"true": "More than 10 days overdue", "false": "10 or fewer days overdue"},
        }},
        "expected": {"escalate": True},
    },
    {
        "name": "Boolean rubric: exact threshold does not qualify",
        "state": "The invoice is 10 days overdue.",
        "questions": {"escalate": {
            **noul("Should this invoice be escalated under the rubric?"),
            "criteria": {"true": "More than 10 days overdue", "false": "10 or fewer days overdue"},
        }},
        "expected": {"escalate": False},
    },
    {
        "name": "Evidence: missing information is not confirmation",
        "state": "A package was shipped yesterday. No delivery confirmation is available.",
        "questions": {"confirmed": noul("Does the state explicitly confirm that the package has been delivered?")},
        "expected": {"confirmed": False},
    },
    {
        "name": "Two booleans: reversed question order",
        "state": "The person says: 'Hanna, have you seen Jeff?' Hanna and Jeff are both present.",
        "questions": {
            "jeff": noul("Is the person speaking directly to Jeff rather than merely mentioning him?"),
            "hanna": noul("Is the person speaking directly to Hanna?"),
        },
        "expected": {"jeff": False, "hanna": True},
    },
    {
        "name": "Single boolean: addressee without another question",
        "state": "The person says: 'Hanna, have you seen Jeff?' Hanna and Jeff are both present.",
        "questions": {"hanna": noul("Is the person speaking directly to Hanna?")},
        "expected": {"hanna": True},
    },
    {
        "name": "Routing: duplicate charge",
        "state": "The customer was charged twice for the same order. Login works normally.",
        "questions": {"team": choice("Which team should resolve the reported problem?", {
            "billing": "Payments, invoices, and duplicate charges",
            "technical": "Login failures and software bugs",
            "shipping": "Parcel delivery and tracking",
        })},
        "expected": {"team": "billing"},
    },
    {
        "name": "Routing: technical problem with billing distractor",
        "state": "The customer paid the invoice successfully. The problem is that the app crashes on every login attempt.",
        "questions": {"team": choice("Which team should resolve the current problem?", {
            "billing": "Failed payments and invoice disputes",
            "technical": "Login failures and software bugs",
            "shipping": "Parcel delivery and tracking",
        })},
        "expected": {"team": "technical"},
    },
    {
        "name": "Choice: explicit unknown alternative",
        "state": "The customer asks when the shop opens. No payment method is mentioned.",
        "questions": {"payment": choice("Which payment method did the customer use?", {
            "card": "A card payment is explicitly stated",
            "cash": "A cash payment is explicitly stated",
            "unknown": "The payment method is not stated",
        })},
        "expected": {"payment": "unknown"},
    },
    {
        "name": "Choice: null descriptions use option keys",
        "state": "The sky color in the supplied observation is blue.",
        "questions": {"color": choice("What is the observed sky color?", {
            "red": None, "blue": None, "green": None,
        })},
        "expected": {"color": "blue"},
    },
    {
        "name": "Choice: custom IDs are preserved",
        "state": "A parcel has not arrived and its tracking number no longer works.",
        "questions": {"ticket.route/v1": choice("Which category matches this issue?", {
            "team_billing": "Payments and invoices",
            "team_delivery": "Parcel delivery and tracking",
            "team_accounts": "Account access",
        })},
        "expected": {"ticket.route/v1": "team_delivery"},
    },
    {
        "name": "Structured: nested object state",
        "state": {"order": {"id": "A17", "payment": {"status": "paid"}, "shipment": {"status": "pending"}}},
        "questions": {
            "paid": noul("Is order A17 marked as paid?"),
            "shipped": noul("Is order A17 marked as shipped?"),
        },
        "expected": {"paid": True, "shipped": False},
    },
    {
        "name": "Structured: array state",
        "state": [{"item": "apple", "stock": 0}, {"item": "pear", "stock": 5}],
        "questions": {"available": choice("Which item has stock greater than zero?", {
            "apple": None, "pear": None,
        })},
        "expected": {"available": "pear"},
    },
    {
        "name": "Structured: object instructions",
        "state": {"temperature_c": 35},
        "questions": {"hot": noul({
            "task": "Determine whether the temperature is hot",
            "rule": "Hot means temperature_c is greater than 30",
        })},
        "expected": {"hot": True},
    },
    {
        "name": "Structured: array instructions",
        "state": "Available stock: 0 units.",
        "questions": {"status": choice([
            "Read the available stock.",
            "Choose empty for zero units and available for one or more units.",
        ], {"empty": "Zero units", "available": "At least one unit"})},
        "expected": {"status": "empty"},
    },
    {
        "name": "Unicode: names and non-ASCII option keys",
        "state": "The confirmed recipient is Zoë. José is the sender.",
        "questions": {"recipient": choice("Who is the confirmed recipient?", {"Jose": None, "Zoe": None})},
        "expected": {"recipient": "Zoe"},
    },
    {
        "name": "Score: two-level minimum scale",
        "state": "The task status is complete.",
        "questions": {"progress": score("Select the level matching the task status.", ["Incomplete", "Complete"])},
        "expected": {"progress": 1},
    },
    {
        "name": "Score: ten-level maximum scale",
        "state": "The recorded level is exactly 9.",
        "questions": {"level": score("Select the recorded level.", [f"Level {i}" for i in range(10)])},
        "expected": {"level": 9},
    },
    {
        "name": "Mixed: boolean, choice, and score in one request",
        "state": "Service: checkout. Status: offline. All customers are affected. Route checkout failures to payments. Urgency is high when all customers are affected.",
        "questions": {
            "online": noul("Is checkout online?"),
            "team": choice("Where should this incident be routed?", {
                "accounts": "Account access", "payments": "Checkout failures", "shipping": "Delivery delays",
            }),
            "urgency": score("Select urgency using the state rule.", ["Low", "Medium", "High"]),
        },
        "expected": {"online": False, "team": "payments", "urgency": 2},
    },
    {
        "name": "Multiple questions: independent facts and opposing answers",
        "state": "The door is open. The window is closed. The light is on. The fan is off.",
        "questions": {
            "door_open": noul("Is the door open?"),
            "window_open": noul("Is the window open?"),
            "light_on": noul("Is the light on?"),
            "fan_on": noul("Is the fan on?"),
        },
        "expected": {"door_open": True, "window_open": False, "light_on": True, "fan_on": False},
    },
])


# Move the correct answer through every position, including the previously
# untested first and last positions (reversing three options leaves the middle).
for keys in [("four", "three", "five"), ("three", "five", "four")]:
    CASES.append({
        "name": f"Choice position: correct answer {'first' if keys[0] == 'four' else 'last'}",
        "state": "The result of 2 + 2 is 4.",
        "questions": {"result": choice("Which option gives the correct result of 2 + 2?", {
            key: {"three": "3", "four": "4", "five": "5"}[key] for key in keys
        })},
        "expected": {"result": "four"},
    })

for value, expected in [(0, 0), (3, 0), (4, 1), (7, 1), (8, 2), (10, 2)]:
    CASES.append({
        "name": f"Score boundary: observed value {value}",
        "state": f"Observed number: {value}.",
        "questions": {"range": score("Select the inclusive range containing the observed number.", [
            "Low: 0 to 3", "Medium: 4 to 7", "High: 8 to 10",
        ])},
        "expected": {"range": expected},
    })

# Exercise both sides of the 26-label / binary-ranking implementation boundary.
for count in (26, 27):
    CASES.append({
        "name": f"Choice boundary: {count} options",
        "state": f"The selected item number is {count}.",
        "questions": {"item": choice(f"The selected item number is {count}. Choose the correspondent item with the selected number.", {
            f"item_{i}": f"Item number {i}" for i in range(1, count + 1)
        })},
        "expected": {"item": f"item_{count}"},
    })


# Sample valid target positions reproducibly across several option counts.
choice_random = random.Random(2026)
for count in (26, 27, 20, 50, 15, 65):
    correct_question = choice_random.randint(1, count)
    CASES.append({
        "name": f"Choice sampled: {count} options, item {correct_question}",
        "state": f"The number is {correct_question}.",
        "questions": {"item": choice(f"The number is {correct_question}. Choose the correspondent item with the selected number.", {
            f"item_{i}": f"Item {i}" for i in range(1, count + 1)
        })},
        "expected": {"item": f"item_{correct_question}"},
    })



def request_api(url, token, payload, timeout):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/systemone",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:1200]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection failed: {exc.reason}") from exc
    return json.loads(content), status, (time.perf_counter() - started) * 1000


def show_result(question_id, question, answer, expected):
    kind = question["type"]
    if answer.get("type") != kind:
        raise ValueError(f"{question_id}: unexpected type: {answer!r}")

    if kind == "noul":
        p = float(answer["noul"])
        if not 0 <= p <= 1:
            raise ValueError(f"{question_id}: noul outside [0,1]: {p}")
        predicted = p >= 0.5
        brier = (p - float(expected)) ** 2
        print(
            f"  {question_id}: noul={p:.4f} | pred={predicted} | "
            f"expected={expected} | {'OK' if predicted == expected else 'FAIL'} | "
            f"Brier={brier:.4f}"
        )
        return predicted == expected, brier

    if kind == "choice":
        probabilities = answer["probabilities"]
        if set(probabilities) != set(question["criteria"]):
            raise ValueError(f"{question_id}: IDs in probabilities do not match criteria")
        total = sum(float(p) for p in probabilities.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"{question_id}: probabilities sum to {total}, not 1")
        if not all(0 <= float(p) <= 1 for p in probabilities.values()):
            raise ValueError(f"{question_id}: probability outside [0,1]")
        selected = answer["choice"]
        confidence = float(answer["confidence"])
        top = max(probabilities, key=probabilities.get)
        if selected != top or abs(confidence - float(probabilities[top])) > 0.001:
            raise ValueError(f"{question_id}: choice/confidence inconsistent with probabilities")
        print(
            f"  {question_id}: choice={selected!r} | expected={expected!r} | "
            f"adapter_confidence={confidence:.4f} | "
            f"distribution={json.dumps(probabilities, ensure_ascii=False)} | "
            f"{'OK' if selected == expected else 'FAIL'}"
        )
        return selected == expected, None

    probabilities = answer["probabilities"]
    values = [float(probabilities[str(i)]) for i in range(len(question["criteria"]))]
    if not all(0 <= p <= 1 for p in values) or abs(sum(values) - 1) > 0.001:
        raise ValueError(f"{question_id}: invalid score distribution: {values}")
    actual_score = float(answer["score"])
    weighted_score = sum(i * p for i, p in enumerate(values))
    confidence = float(answer["confidence"])
    if abs(actual_score - weighted_score) > 0.001 or abs(confidence - max(values)) > 0.001:
        raise ValueError(f"{question_id}: score/confidence inconsistent with probabilities")
    predicted_level = max(range(len(values)), key=lambda i: values[i])
    print(
        f"  {question_id}: selected_level={predicted_level} | "
        f"expected={expected} | weighted_score={actual_score:.4f} | "
        f"adapter_confidence={confidence:.4f} | distribution={values} | "
        f"{'OK' if predicted_level == expected else 'FAIL'}"
    )
    return predicted_level == expected, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--token", default=os.getenv("LOCAL_API_KEY", ""), help="Server API key (defaults to LOCAL_API_KEY)")
    parser.add_argument("--repeat", type=int, default=1, help="Repetitions of each case")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--case", action="append", default=[], help="Run names containing this text (case-insensitive; repeatable, matches any)")
    parser.add_argument("--list", action="store_true", help="List matching cases without contacting the server")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    cases = [case for case in CASES if not args.case or any(
        fragment.casefold() in case["name"].casefold() for fragment in args.case
    )]
    if not cases:
        parser.error("No cases match --case")
    if args.list:
        for case in cases:
            print(f"{case['name']} ({len(case['questions'])} decisions)")
        print(f"Total: {len(cases)} cases, {sum(len(case['questions']) for case in cases)} decisions per run")
        return

    correct = total = contract_errors = 0
    failures = []
    timings = []
    briers = []
    for run in range(1, args.repeat + 1):
        print(f"\n========== RUN {run}/{args.repeat} ==========")
        for case in cases:
            payload = {"model": "jev-latest", "state": case["state"], "questions": case["questions"]}
            print(f"\n{case['name']}")
            try:
                response, status, elapsed_ms = request_api(args.url, args.token, payload, args.timeout)
                if status != 200:
                    raise ValueError(f"Unexpected status: {status}")
                timings.append(elapsed_ms)
                print(f"  HTTP {status} | {elapsed_ms:.1f} ms | model={response.get('model')}")
                answers = response["answers"]
                if set(answers) != set(case["questions"]):
                    raise ValueError("Returned IDs differ from question IDs")
                for qid, question in case["questions"].items():
                    is_correct, brier = show_result(qid, question, answers[qid], case["expected"][qid])
                    total += 1
                    correct += is_correct
                    if not is_correct:
                        failures.append(f"Run {run}: {case['name']} / {qid}")
                    if brier is not None:
                        briers.append(brier)
            except (RuntimeError, ValueError, KeyError, TypeError) as exc:
                contract_errors += 1
                failures.append(f"Run {run}: {case['name']} / CONTRACT/HTTP ERROR")
                print(f"  CONTRACT/HTTP ERROR: {exc}")

    print("\n========== SUMMARY ==========")
    print(f"Correct decisions: {correct}/{total} | failed requests: {contract_errors}")
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  - {failure}")
    if timings:
        print(f"Mean HTTP latency: {statistics.mean(timings):.1f} ms | median: {statistics.median(timings):.1f} ms")
    if briers:
        print(f"Mean boolean Brier score: {statistics.mean(briers):.4f} (small sample; does NOT demonstrate calibration)")
    print("Note: confidence=1.0 and one-hot probabilities encode discrete decisions; they do not measure model certainty.")
    if contract_errors or correct != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
