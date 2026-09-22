# SPDX-License-Identifier: Apache-2.0
"""Contract tests without model downloads or GPU inference."""

import contextlib
import io
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from maskdecide import api
import test_jev_api_smoke as smoke


class DecisionContractTests(unittest.TestCase):
    def test_mixed_questions_keep_ids_and_discrete_values(self):
        request = api.JevRequest(
            state={"message": "Duplicate charge"},
            questions={
                "needs_help": {"type": "noul", "instructions": "Needs help?"},
                "team": {
                    "type": "choice", "instructions": "Choose a team",
                    "criteria": {"billing": "Invoices", "support": "Login"},
                },
                "urgency": {
                    "type": "score", "instructions": "Rate urgency",
                    "criteria": ["Low", "Medium", "High"],
                },
            },
        )
        api.validate_jev_request(request)
        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            prompt, slots, plans = api.build_jev(request)
        self.assertIn('"message":"Duplicate charge"', prompt)
        self.assertEqual([slot.allowed for slot in slots], ["AB", "AB", "ABC"])
        answers = api.create_jev_answers(plans, ["B", "A", "C"], [[], [], []], [[], [], []])
        self.assertEqual(set(answers), set(request.questions))
        self.assertEqual(answers["needs_help"], {"type": "noul", "noul": 0.0})
        self.assertEqual(answers["team"]["choice"], "billing")
        self.assertEqual(answers["team"]["probabilities"], {"billing": 1.0, "support": 0.0})
        self.assertEqual(answers["urgency"]["score"], 2.0)
        self.assertEqual(answers["urgency"]["legend"], {"0": "Low", "1": "Medium", "2": "High"})
        self.assertEqual(answers["urgency"]["confidence"], 1.0)

    def test_large_choice_uses_margin_ranking(self):
        request = api.JevRequest(state="Context", questions={"winner": {
            "type": "choice", "instructions": "Choose the best option",
            "criteria": {f"option_{i}": None for i in range(27)},
        }})
        api.validate_jev_request(request)
        with patch.object(api, "format_prompt", return_value="prompt"):
            _, slots, plans = api.build_jev(request)
        self.assertEqual(len(slots), 27)
        # Every slot says yes. The winner must use the margin, not the first
        # yes or the largest absolute yes logit.
        raw = [[10.0, 9.0] for _ in slots]
        raw[26] = [3.0, -1.0]
        answer = api.create_jev_answers(plans, ["A"] * 27, raw, [[] for _ in slots])["winner"]
        self.assertEqual(answer["choice"], "option_26")
        self.assertEqual(sum(answer["probabilities"].values()), 1.0)

    def test_invalid_criteria_are_rejected(self):
        cases = [
            ("choice", {"only": None}),
            ("choice", {str(i): None for i in range(256)}),
            ("score", ["only"]),
            ("score", list(range(11))),
            ("noul", {"maybe": "Uncertain"}),
            ("choice", {"1invalid": "Starts with digit"}),
            ("choice", {"has space": "Contains space"}),
            ("choice", {"special!": "Punctuation"}),
        ]
        for kind, criteria in cases:
            with self.subTest(kind=kind, criteria=criteria):
                request = api.JevRequest(state="Context", questions={"q": {
                    "type": kind, "instructions": "Decide", "criteria": criteria,
                }})
                with self.assertRaises(HTTPException) as error:
                    api.validate_jev_request(request)
                self.assertEqual(error.exception.status_code, 422)

    def test_isolated_builds_one_prompt_per_question(self):
        request = api.JevRequest(
            state="The customer was charged twice.",
            isolated=True,
            questions={
                "billing": {"type": "noul", "instructions": "Is this a billing issue?"},
                "team": {"type": "choice", "instructions": "Choose a team", "criteria": {"billing": "Invoices", "support": "Login"}},
            },
        )
        api.validate_jev_request(request)
        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            isolated = api.build_jev_isolated(request)
        self.assertEqual(len(isolated), 2)
        self.assertEqual([plan.question_id for _, _, plan in isolated], ["billing", "team"])
        self.assertEqual([len(slots) for _, slots, _ in isolated], [1, 1])
        self.assertIn("charged twice", isolated[0][0])
        self.assertIn("charged twice", isolated[1][0])
        self.assertNotIn("Question 2", isolated[0][0])
        self.assertNotIn("Question 1", isolated[1][0])

    def test_local_options_must_be_unique(self):
        request = api.DecisionRequest(question="Pick", type="single_choice", options=[
            api.Option(id="same", text="First"), api.Option(id="same", text="Second"),
        ])
        with self.assertRaises(HTTPException) as error:
            api.validate_local_request(request)
        self.assertEqual(error.exception.detail, "Option IDs must be unique")

    def test_boolean_rejects_options(self):
        request = api.DecisionRequest(question="True?", type="boolean", options=[
            api.Option(id="unexpected", text="Unexpected"),
        ])
        with self.assertRaises(HTTPException) as error:
            api.validate_local_request(request)
        self.assertEqual(error.exception.status_code, 422)

    def test_optional_authentication(self):
        with patch.object(api, "LOCAL_API_KEY", "example-key"):
            api.require_local_key("Bearer example-key")
            for authorization in [None, "Bearer wrong", "example-key"]:
                with self.subTest(authorization=authorization):
                    with self.assertRaises(HTTPException) as error:
                        api.require_local_key(authorization)
                    self.assertEqual(error.exception.status_code, 401)
        with patch.object(api, "LOCAL_API_KEY", None):
            api.require_local_key(None)

    def test_literal_mask_in_prompt_or_slot_is_a_client_error(self):
        for encodings in [[[api.MASK_ID]], [[5], [api.MASK_ID]]]:
            with self.subTest(encodings=encodings), patch.object(api, "tokenizer") as tokenizer:
                tokenizer.encode.side_effect = encodings
                with self.assertRaises(HTTPException) as error:
                    api.build_masked_input("prompt", [api.Slot("label", "AB")])
                self.assertEqual(error.exception.status_code, 422)


class SmokeExitTests(unittest.TestCase):
    def run_smoke(self, value):
        case = {"name": "Boolean", "state": "True", "questions": {
            "q": smoke.noul("True?"),
        }, "expected": {"q": True}}
        response = {"model": "test", "answers": {"q": {"type": "noul", "noul": value}}}
        with patch.object(smoke, "CASES", [case]), \
             patch.object(smoke, "request_api", return_value=(response, 200, 1.0)), \
             patch("sys.argv", ["test_jev_api_smoke.py"]), \
             contextlib.redirect_stdout(io.StringIO()):
            smoke.main()

    def test_wrong_decision_exits_with_failure(self):
        with self.assertRaises(SystemExit) as error:
            self.run_smoke(0.0)
        self.assertEqual(error.exception.code, 1)

    def test_correct_decision_succeeds(self):
        self.run_smoke(1.0)

    def test_invalid_response_exits_with_failure(self):
        with self.assertRaises(SystemExit) as error:
            self.run_smoke(2.0)
        self.assertEqual(error.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
