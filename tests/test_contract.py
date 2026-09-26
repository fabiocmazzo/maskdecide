# SPDX-License-Identifier: Apache-2.0
"""Contract tests without model downloads or GPU inference."""

import asyncio
import contextlib
import io
import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from fastapi import HTTPException

from maskdecide import api
from maskdecide.decoding import decode
try:
    import test_jev_api_smoke as smoke
except ImportError:  # tests/ layout: smoke module moved under tests/
    from tests import test_jev_api_smoke as smoke


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
        self.assertIn('"message":"Duplicate charge"', prompt.text)
        self.assertEqual([slot.allowed for slot in slots], [["A", "B"], ["A", "B"], ["A", "B", "C"]])
        answers = api.create_jev_answers(plans, ["B", "A", "C"], [[], [], []], [[], [], []])
        self.assertEqual(set(answers), set(request.questions))
        self.assertEqual(answers["needs_help"], {"type": "noul", "noul": 0.0})
        self.assertEqual(answers["team"]["choice"], "billing")
        self.assertEqual(answers["team"]["probabilities"], {"billing": 1.0, "support": 0.0})
        self.assertEqual(answers["urgency"]["score"], 2.0)
        self.assertEqual(answers["urgency"]["legend"], {"0": "Low", "1": "Medium", "2": "High"})
        self.assertEqual(answers["urgency"]["confidence"], 1.0)

    def test_large_choice_uses_one_mask_with_pair_labels(self):
        request = api.JevRequest(state="Context", questions={"winner": {
            "type": "choice", "instructions": "Choose the best option",
            "criteria": {f"option_{i}": None for i in range(27)},
        }})
        api.validate_jev_request(request)
        with patch.object(api, "CHOICE_BINARY_MIN_OPTIONS", 256), \
             patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            _, slots, plans = api.build_jev(request)
        # 27 options fit on a single mask: A..Z plus the first pair label.
        self.assertEqual(len(slots), 1)
        self.assertEqual(len(plans[0].slot_indices), 1)
        self.assertEqual(slots[0].label, "\nQuestion 1: Answer:")
        labels = api.choice_labels(27)
        self.assertEqual(labels[:26], list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertEqual(labels[26], "AA")
        self.assertEqual(slots[0].allowed, labels)
        answer = api.create_jev_answers(plans, ["AA"], [[]], [[]])["winner"]
        self.assertEqual(answer["choice"], "option_26")
        self.assertEqual(answer["probabilities"]["option_26"], 1.0)
        self.assertEqual(sum(answer["probabilities"].values()), 1.0)

    def test_26th_choice_maps_z_to_last_option(self):
        request = api.JevRequest(state="The selected item number is 26.", questions={"item": {
            "type": "choice", "instructions": "Choose the item with the selected number.",
            "criteria": {f"item_{i}": f"Item number {i}" for i in range(1, 27)},
        }})
        with patch.object(api, "CHOICE_BINARY_MIN_OPTIONS", 256), \
             patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            _, slots, plans = api.build_jev(request)
        self.assertEqual(slots[0].allowed, list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        self.assertEqual(slots[0].label, "\nQuestion 1: Answer:")
        answer = api.create_jev_answers(plans, ["Z"], [[]], [[]])["item"]
        self.assertEqual(answer["choice"], "item_26")

    def test_large_choice_uses_independent_binary_slots_with_same_response(self):
        request = api.JevRequest(state="The selected item number is 17.", questions={"item": {
            "type": "choice", "instructions": "Choose the item with the selected number.",
            "criteria": {f"item_{i}": f"Item number {i}" for i in range(1, 28)},
        }})
        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            prompt, slots, plans = api.build_jev(request)
        self.assertEqual(plans[0].mode, "binary_ranking")
        self.assertEqual(len(slots), 27)
        self.assertEqual([slot.allowed for slot in slots], [["A", "B"]] * 27)
        self.assertEqual(len({slot.group_id for slot in slots}), 27)
        self.assertIn("The selected item number is 17.", prompt.text)
        self.assertIn("Candidate: item_17: Item number 17", prompt.text)
        for slot, (_, end, group_id) in zip(slots, prompt.answer_spans, strict=True):
            self.assertEqual((slot.insertion_offset, slot.group_id), (end, group_id))

        labels = ["B"] * 27
        labels[16] = "A"
        logits = [[0.0, 2.0] for _ in slots]
        logits[16] = [3.0, 0.0]
        answer = api.create_jev_answers(plans, labels, logits, [[0.1, 0.9]] * 27)["item"]
        self.assertEqual(answer["choice"], "item_17")
        self.assertEqual(len(answer["probabilities"]), 27)
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)

    def test_50_option_choice_is_chunked_and_returns_one_choice(self):
        request = api.JevRequest(state="The selected item number is 25.", questions={"item": {
            "type": "choice", "instructions": "Choose the item with the selected number.",
            "criteria": {f"item_{i}": f"Item number {i}" for i in range(1, 51)},
        }})
        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            groups = api.build_jev_groups(request)
        self.assertEqual([len(slots) for _, slots, _ in groups], [17, 17, 17, 17, 16, 16])
        self.assertEqual([plans[0].mode for _, _, plans in groups], ["binary_ranking"] * 6)

        def fake_evaluate(prompt, slots):
            numbers = [int(re.search(r"Candidate: item_(\d+)", prompt.text[start:end]).group(1))
                       for start, end, _ in prompt.answer_spans]
            logits = [[3.0, 0.0] if number == 25 else [0.0, 2.0] for number in numbers]
            labels = ["A" if number == 25 else "B" for number in numbers]
            return labels, logits, [[0.5, 0.5]] * len(slots), 100, 1

        with patch.object(api, "evaluate", side_effect=fake_evaluate):
            answers, tokens, passes = asyncio.run(api.evaluate_jev_groups(groups))
        self.assertEqual(list(answers), ["item"])
        self.assertEqual(answers["item"]["choice"], "item_25")
        self.assertEqual(len(answers["item"]["probabilities"]), 50)
        self.assertEqual((tokens, passes), (600, 6))

    def test_reversed_binary_scoring_cancels_position_bias(self):
        request = api.JevRequest(state="The selected item is 10.", questions={"item": {
            "type": "choice", "instructions": "Choose the selected item.",
            "criteria": {f"item_{i}": f"Item number {i}" for i in range(1, 21)},
        }})
        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            groups = api.build_jev_groups(request)
        self.assertEqual(len(groups), 2)

        def fake_evaluate(prompt, slots):
            numbers = [int(re.search(r"Candidate: item_(\d+)", prompt.text[start:end]).group(1))
                       for start, end, _ in prompt.answer_spans]
            # In either order, late positions beat the correct candidate in
            # this single pass. Averaging both orders cancels that bias.
            margins = [(1.0 if number == 10 else 0.0) + 0.2 * position
                       for position, number in enumerate(numbers)]
            logits = [[margin, 0.0] for margin in margins]
            return ["A"] * len(slots), logits, [[0.5, 0.5]] * len(slots), 100, 1

        with patch.object(api, "evaluate", side_effect=fake_evaluate):
            answers, _, passes = asyncio.run(api.evaluate_jev_groups(groups))
        self.assertEqual(passes, 2)
        self.assertEqual(answers["item"]["choice"], "item_10")

    def test_pairwise_tournament_cancels_second_position_bias(self):
        request = api.JevRequest(state="The selected item is 3.", questions={"item": {
            "type": "choice", "instructions": "Choose the selected item.",
            "criteria": {f"item_{i}": f"Item {i}" for i in range(1, 5)},
        }})

        def fake_evaluate(prompt, slots):
            options = [int(value) for value in re.findall(r"[AB]: item_(\d+)", prompt.text)]
            # The raw A/B logits favor B by two units in either order.
            margin = (2.0 if options[0] == 3 else 0.0) - (2.0 if options[1] == 3 else 0.0) - 2.0
            return ["A" if margin >= 0 else "B"], [[margin, 0.0]], [[0.5, 0.5]], 100, 1

        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)), \
             patch.object(api, "evaluate", side_effect=fake_evaluate):
            answer, tokens, passes = asyncio.run(api.evaluate_pairwise_choice(
                request, "item", request.questions["item"], 1,
            ))
        self.assertEqual(answer["choice"], "item_3")
        self.assertEqual(set(answer["probabilities"]), set(request.questions["item"].criteria))
        self.assertAlmostEqual(sum(answer["probabilities"].values()), 1.0)
        self.assertEqual((tokens, passes), (600, 6))

    def test_system_one_combines_pairwise_and_ordinary_answers_in_request_order(self):
        request = api.JevRequest(state="The selected item is 15.", questions={
            "large": {"type": "choice", "instructions": "Choose item 15.",
                      "criteria": {f"item_{i}": f"Item {i}" for i in range(1, 16)}},
            "plain": {"type": "noul", "instructions": "Is the selected item 15?"},
        })
        async def ordinary_result(groups):
            self.assertEqual(len(groups), 1)
            return {"plain": {"type": "noul", "noul": 0.9}}, 10, 1

        async def pairwise_result(*args):
            return {"type": "choice", "choice": "item_15", "confidence": 1.0,
                    "probabilities": {f"item_{i}": float(i == 15) for i in range(1, 16)}}, 20, 40

        with patch.object(api, "model", object()), \
             patch.object(api, "require_local_key"), \
             patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)), \
             patch.object(api, "evaluate_jev_groups", side_effect=ordinary_result), \
             patch.object(api, "evaluate_pairwise_choice", side_effect=pairwise_result):
            response = asyncio.run(api.system_one(request))
        self.assertEqual(list(response["answers"]), ["large", "plain"])
        self.assertEqual(response["answers"]["large"]["choice"], "item_15")
        self.assertEqual(response["usage"], {"input_tokens": 30, "output_tokens": 0, "forward_passes": 41})

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

    def test_evaluate_logs_masked_and_filled_assistant_text(self):
        mask = 99
        slots = [api.Slot("Answer 1:", ["A", "B"], 1), api.Slot("Answer 2:", ["A", "B"], 2)]
        tokens = torch.tensor([[10, mask, 11, mask]])
        vocabulary = {10: "Question 1 Answer 1:", 11: " Question 2 Answer 2:",
                      mask: "<mask>", 20: " A", 21: " B"}
        fake_tokenizer = SimpleNamespace(
            decode=lambda ids, **_: "".join(vocabulary[token] for token in ids),
        )
        decoded = SimpleNamespace(winners=[1, 0], logits=[[0.0, 2.0], [3.0, 1.0]],
                                  probabilities=[[0.12, 0.88], [0.88, 0.12]], forward_passes=1)
        with patch.object(api, "tokenizer", fake_tokenizer), \
             patch.object(api, "build_masked_input", return_value=(tokens, [1, 3], None)), \
             patch.object(api, "letter_token_ids", return_value={"A": 20, "B": 21}), \
             patch.object(api, "decode", return_value=decoded), \
             patch.object(api, "gpu_stats"), \
             patch.object(api.torch.cuda, "reset_peak_memory_stats"), \
             patch.object(api.logger, "info") as log:
            labels, _, _, _, _ = api.evaluate("prompt", slots)

        self.assertEqual(labels, ["B", "A"])
        messages = {call.args[0]: call.args[1] for call in log.call_args_list
                    if call.args[0] in {"PROMPT | %s", "ASSISTANT INPUT | %s",
                                        "SLOTS | %s", "ASSISTANT RESULT | %s"}}
        self.assertEqual(messages["PROMPT | %s"], "prompt")
        self.assertEqual(messages["ASSISTANT INPUT | %s"],
                         "Question 1 Answer 1:<mask> Question 2 Answer 2:<mask>")
        self.assertEqual(messages["ASSISTANT RESULT | %s"],
                         "Question 1 Answer 1: B Question 2 Answer 2: A")
        self.assertIn('"selected": "B"', messages["SLOTS | %s"])
        self.assertIn('"group_id": 2', messages["SLOTS | %s"])
        self.assertIn('"logits": {"A": 0.0, "B": 2.0}', messages["SLOTS | %s"])

    def test_boolean_slot_asks_for_answer_without_reasking_yes(self):
        request = api.JevRequest(state="2 + 2 = 4", questions={
            "is_five": {"type": "noul", "instructions": "Is 2 + 2 equal to 5?"},
        })
        with patch.object(api, "format_prompt", side_effect=lambda parts: "\n".join(parts)):
            _, slots, _ = api.build_jev(request)
        self.assertEqual(slots[0].label, "\nQuestion 1: Answer:")
        self.assertEqual(slots[0].allowed, ["A", "B"])

    def test_boolean_response_uses_probability_of_a(self):
        plan = api.QuestionPlan("q", "noul", [0], [], [], "binary")
        answer = api.create_jev_answers([plan], ["B"], [[0.0, 2.0]], [[0.12, 0.88]])
        self.assertEqual(answer["q"], {"type": "noul", "noul": 0.12})

    def test_candidate_projection_keeps_small_bfloat16_score_difference(self):
        class ProjectionModel:
            def __init__(self):
                self.diffusion_head = torch.nn.Linear(1, 3, dtype=torch.bfloat16)
                with torch.no_grad():
                    self.diffusion_head.weight[1, 0] = 1.0
                    self.diffusion_head.weight[2, 0] = 1.0078125
                    self.diffusion_head.bias[1:3] = 10.0

            def __call__(self, **kwargs):
                return SimpleNamespace(last_hidden_state=torch.ones((1, 1, 1), dtype=torch.bfloat16))

        result = decode(ProjectionModel(), torch.tensor([[99]]), [0], [[1, 2]], mask_id=99)
        self.assertEqual(result.winners, [1])
        self.assertGreater(result.logits[0][1], result.logits[0][0])


class SmokeExitTests(unittest.TestCase):
    def test_sampled_choice_targets_are_available(self):
        for case in smoke.CASES:
            if not case["name"].startswith("Choice sampled:"):
                continue
            self.assertIn(case["expected"]["item"], case["questions"]["item"]["criteria"])

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
