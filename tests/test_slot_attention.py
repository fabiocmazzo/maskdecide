"""CPU regression tests; no checkpoint download and no bitsandbytes required."""

import ast
import asyncio
import json
import logging
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import torch
from fastapi import HTTPException
from pydantic import BaseModel, Field
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from maskdecide.decoding import decode
from maskdecide.slot_attention import (
    BACKEND_NAME, build_slot_attention_mask, install_slot_attention,
    slot_attention_scope, verify_model_isolation,
)


class Ministral3Attention(torch.nn.Module):
    """Small test double for the inspected model's attention dispatch contract.

    Uses the real Transformers registry and real PyTorch SDPA, but it is not
    the NVIDIA checkpoint. In particular, this double does not implement RoPE.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.diffusion_lm = True
        self.scaling = 0.5
        self.num_key_value_groups = 2
        self.is_causal = False
        self.q_proj = torch.nn.Linear(8, 8, bias=False)
        self.k_proj = torch.nn.Linear(8, 4, bias=False)
        self.v_proj = torch.nn.Linear(8, 4, bias=False)
        self.o_proj = torch.nn.Linear(8, 8, bias=False)

    def forward(self, hidden):
        batch, length, _ = hidden.shape
        q = self.q_proj(hidden).view(batch, length, 2, 4).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, length, 1, 4).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, length, 1, 4).transpose(1, 2)
        # Intentionally discard any caller mask, like the published diffusion path.
        output, _ = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation](
            self, q, k, v, None, dropout=0.0, scaling=self.scaling, is_causal=False,
        )
        return self.o_proj(output.reshape(batch, length, 8))


class SmallModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="sdpa")
        self.encoder = torch.nn.Module()
        self.encoder.embed_tokens = torch.nn.Embedding(256, 8)
        self.encoder.layers = torch.nn.ModuleList()
        for _ in range(2):
            layer = torch.nn.Module()
            layer.self_attn = Ministral3Attention(self.config)
            self.encoder.layers.append(layer)
        self.diffusion_head = torch.nn.Linear(8, 256, bias=True)
        self.eval()

    def get_input_embeddings(self):
        return self.encoder.embed_tokens

    def forward(self, input_ids, **kwargs):
        hidden = self.encoder.embed_tokens(input_ids)
        for layer in self.encoder.layers:
            hidden = hidden + layer.self_attn(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


def api_namespace(model):
    """Load actual pure API builders without executing its CUDA server imports."""
    path = Path(__file__).parents[1] / "src/maskdecide/api.py"
    tree = ast.parse(path.read_text())
    names = {
        "JevQuestion", "JevRequest", "Slot", "QuestionPlan", "TaggedPrompt",
        "Option", "DecisionRequest", "build_local",
        "as_text", "format_prompt", "build_jev", "tag_jev_prompt", "choice_labels",
        "validate_group_size", "build_jev_groups", "build_masked_input", "token_question_group",
        "validate_decoded_results", "rank_binary_choice", "create_jev_answers", "evaluate_jev_groups",
    }
    nodes = [n for n in tree.body if getattr(n, "name", None) in names]
    tokenizer = Tokenizer(models.WordLevel(
        {"[UNK]": 0, "[MASK]": 255, "A": 1, "B": 2, "Hanna": 3, "Jeff": 4},
        unk_token="[UNK]",
    ))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", mask_token="[MASK]")
    fast.chat_template = "USER: {{ messages[0]['content'] }} END ASSISTANT:"
    real_encode = fast.encode
    def label_aware_encode(text, add_special_tokens=False):
        # Two-letter pair labels count as one token, mirroring the real
        # tokenizer; everything else uses the small WordLevel vocab.
        token = text.strip()
        if len(token) == 2 and token.isalpha() and token.isupper():
            return [200 + (ord(token[0]) - 65) * 26 + (ord(token[1]) - 65)]
        return real_encode(text, add_special_tokens=add_special_tokens)
    fast.encode = label_aware_encode
    ns = dict(
        globals(), MAX_QUESTIONS=128, MAX_QUESTIONS_PER_PROMPT=3,
        MAX_INPUT_TOKENS=4096, LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        MAX_CHOICE_OPTIONS=255, CHOICE_BINARY_MIN_OPTIONS=20,
        BINARY_CHOICE_CHUNK_SIZE=20, MAX_LOCAL_OPTIONS=20,
        USE_SLOT_ATTENTION=True, MASK_ID=255, model=model, tokenizer=fast,
        DECODER_MODE="one_pass", logger=logging.getLogger("test"),
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec"), ns)
    return ns


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.model = SmallModel()
        install_slot_attention(self.model)
        self.ids = torch.tensor([[1, 2, 3, 4, 255, 5, 6, 255]])
        self.groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2])

    def test_mask_direction_and_validation(self):
        matrix = build_slot_attention_mask(torch.tensor([0, 1, 2]))[0, 0]
        # Only question->other-question attention is blocked; the shared
        # state row sees everything so its representations stay intact.
        self.assertEqual(matrix.tolist(), [[True, True, True], [True, True, False], [True, False, True]])
        for bad in [torch.tensor([-1, 0]), torch.tensor([1, 2]), torch.tensor([0.0, 1.0])]:
            with self.assertRaises(ValueError):
                build_slot_attention_mask(bad)

    def test_isolation_across_two_layers_and_state_influence(self):
        result = verify_model_isolation(self.model, [1, 2, 3, 4, 5, 6], 255)
        # The state sees every question (state_visibility), and the state
        # influences both answer slots (shared_state_slot_deltas).
        self.assertIn("cross_question_slot_delta", result)
        self.assertGreater(result["state_visibility_delta"], 0.0)
        self.assertTrue(all(d > 0 for d in result["shared_state_slot_deltas"]))
        self.assertEqual(self.model.config._attn_implementation, "sdpa")
        self.assertEqual(self.model.encoder.layers[0].self_attn.config._attn_implementation, BACKEND_NAME)

    def test_no_scope_retains_original_sdpa(self):
        tokens = self.ids
        actual = self.model(tokens).last_hidden_state
        for layer in self.model.encoder.layers:
            layer.self_attn.config._attn_implementation = "sdpa"
        expected = self.model(tokens).last_hidden_state
        torch.testing.assert_close(actual, expected)

    def test_reject_missing_layer_and_clear_scope_after_failure(self):
        allowed = build_slot_attention_mask(self.groups)
        with self.assertRaisesRegex(RuntimeError, "every encoder layer"):
            with slot_attention_scope(self.model, allowed):
                pass
        with self.assertRaisesRegex(RuntimeError, "deliberate"):
            with slot_attention_scope(self.model, allowed):
                raise RuntimeError("deliberate")
        with slot_attention_scope(self.model, allowed):
            self.model(self.ids)

    def test_restricted_projection_matches_full_head(self):
        allowed = build_slot_attention_mask(self.groups)
        with slot_attention_scope(self.model, allowed):
            hidden = self.model(self.ids).last_hidden_state
        full = self.model.diffusion_head(hidden)
        original = self.ids.clone()
        result = decode(self.model, self.ids, [4, 7], [[3, 2], [5, 1, 6]], mask_id=255, group_ids=self.groups)
        torch.testing.assert_close(torch.tensor(result.logits[0]), full[0, 4, [3, 2]])
        torch.testing.assert_close(torch.tensor(result.logits[1]), full[0, 7, [5, 1, 6]])
        self.assertEqual(result.forward_passes, 1)
        torch.testing.assert_close(self.ids, original)

    def test_iterative_scheduling_progress_per_question(self):
        with torch.no_grad():
            self.model.diffusion_head.weight.zero_()
            self.model.diffusion_head.bias.zero_()
        ids = torch.tensor([[1, 2, 255, 255, 3, 255]])
        groups = torch.tensor([0, 1, 1, 1, 2, 2])
        args = dict(mask_id=255, mode="iterative", threshold=0.9)
        structured = decode(self.model, ids, [2, 3, 5], [[1, 2]] * 3, group_ids=groups, **args)
        original = decode(self.model, ids, [2, 3, 5], [[1, 2]] * 3, **args)
        self.assertEqual(structured.forward_passes, 2)
        self.assertEqual(original.forward_passes, 3)
        self.assertEqual(structured.probabilities, [[0.5, 0.5]] * 3)

    def test_slot_allowed_in_state_group_for_score(self):
        # Score questions deliberately keep full visibility: their slots sit
        # in the shared-state group 0 and must not be rejected.
        self.groups[4] = 0
        result = decode(self.model, self.ids, [4, 7], [[1, 2]] * 2, mask_id=255, group_ids=self.groups)
        self.assertEqual(len(result.winners), 2)
        self.groups[4] = -1
        with self.assertRaisesRegex(ValueError, "Negative group IDs"):
            decode(self.model, self.ids, [4, 7], [[1, 2]] * 2, mask_id=255, group_ids=self.groups)

    def test_merged_question_boundary_token(self):
        ns = api_namespace(self.model)
        text = "One\n\nTwo"
        spans = [(0, 4, 1), (4, 8, 2)]
        assign = ns["token_question_group"]
        self.assertEqual(assign(text, 3, 5, spans), 2)  # Both newlines.
        self.assertEqual(assign(text, 3, 8, spans), 2)  # Newlines + Two.
        self.assertEqual(assign(text, 0, 5, spans), 1)  # One + newlines.
        self.assertEqual(assign(text, 0, 0, spans), 0)  # Empty special offset.
        with self.assertRaisesRegex(ValueError, "non-whitespace"):
            assign(text, 0, 8, spans)  # Genuine mixed content must not leak.

        class BoundaryTokenizer:
            def __call__(self, *args, **kwargs):
                return {"input_ids": [21, 22, 23], "offset_mapping": [(0, 3), (3, 5), (5, 8)]}

            def encode(self, text, **kwargs):
                return [21, 22, 23] if text == "One\n\nTwo" else [24]

        ns["tokenizer"] = BoundaryTokenizer()
        prompt = ns["TaggedPrompt"](text, spans)
        slots = [ns["Slot"]("A:", ["A", "B"], 1), ns["Slot"]("B:", ["A", "B"], 2)]
        ids, positions, groups = ns["build_masked_input"](prompt, slots)
        self.assertEqual(ids[0, :3].tolist(), [21, 22, 23])
        self.assertEqual(groups[:3].tolist(), [1, 2, 2])
        self.assertEqual(groups[positions].tolist(), [1, 2])
        self.assertGreater(groups[1].item(), 0)  # Separator is never shared.

    def test_token_group_alignment_and_disabled_ab_path(self):
        ns = api_namespace(self.model)
        req = ns["JevRequest"](state={"text":"Question 1: café 😀 Hanna Jeff"}, questions={
            "hanna": {"type":"noul", "instructions":"Hanna?"},
            "intent": {"type":"choice", "instructions":"Intent?", "criteria":{"a":"greeting", "b":"farewell", "c":"request"}},
            "large": {"type":"choice", "instructions":"Large?", "criteria":{f"o{i}":str(i) for i in range(30)}},
        })
        prompt, slots, plans = ns["build_jev"](req)
        ids, positions, groups = ns["build_masked_input"](prompt, slots)
        # Short questions use one slot; each large-choice option has its own
        # yes/no slot and private question group.
        self.assertEqual(groups[positions].tolist(), list(range(1, 33)))
        # Each question's complete token group ends at its own answer mask.
        for index, position in enumerate(positions):
            group = index + 1
            own_positions = (groups == group).nonzero().flatten().tolist()
            self.assertEqual(own_positions[-1], position)
            if index:
                self.assertTrue((groups[positions[index - 1] + 1:position + 1] == group).all())
        self.assertEqual(positions[-1], ids.shape[1] - 1)  # Open assistant ends at its mask.
        ns["USE_SLOT_ATTENTION"] = False
        plain_ids, plain_positions, plain_groups = ns["build_masked_input"](prompt, slots)
        torch.testing.assert_close(ids, plain_ids)
        self.assertEqual(positions, plain_positions)
        self.assertIsNone(plain_groups)
        ns["MAX_INPUT_TOKENS"] = 3
        with self.assertRaises(HTTPException) as error:
            ns["build_masked_input"](prompt, slots)
        self.assertEqual(error.exception.status_code, 422)

    def test_local_answer_slots_follow_question_and_share_attention(self):
        ns = api_namespace(self.model)
        for kind in ("boolean", "single_choice", "multiple_choice"):
            options = [] if kind == "boolean" else [{"id": "a", "text": "First"}, {"id": "b", "text": "Second"}]
            req = ns["DecisionRequest"](context="Shared", question="Select?", type=kind, options=options)
            prompt, slots = ns["build_local"](req)
            ids, positions, groups = ns["build_masked_input"](prompt, slots)
            self.assertEqual(groups[positions].tolist(), [1] * len(slots))
            self.assertEqual(positions[-1], ids.shape[1] - 1)
            self.assertEqual((groups == 1).nonzero()[-1].item(), positions[-1])

    def test_slots_are_in_open_assistant_and_user_copies_share_groups(self):
        ns = api_namespace(self.model)
        ns["tokenizer"].chat_template = (
            "<user>{{ messages[0]['content'] }}</user>"
            "{% if add_generation_prompt %}<assistant>{% endif %}"
        )
        request = ns["JevRequest"](state="Shared facts", questions={
            "a": {"type": "noul", "instructions": "First question?"},
            "b": {"type": "score", "instructions": "Second question?", "criteria": ["low", "high"]},
        })
        prompt, slots, _ = ns["build_jev"](request)
        assistant_start = prompt.text.index("<assistant>") + len("<assistant>")
        self.assertEqual(len(prompt.question_spans), 4)
        for user_span, answer_span, slot in zip(prompt.question_spans[:2], prompt.answer_spans, slots):
            self.assertLess(user_span[1], assistant_start)
            self.assertGreaterEqual(answer_span[0], assistant_start)
            self.assertEqual(user_span[2], answer_span[2])
            self.assertEqual(prompt.text[user_span[0]:user_span[1]], prompt.text[answer_span[0]:answer_span[1]])
            self.assertEqual(slot.insertion_offset, answer_span[1])
        ids, positions, groups = ns["build_masked_input"](prompt, slots)
        self.assertEqual(positions[-1], ids.shape[1] - 1)
        self.assertEqual(groups[positions].tolist(), [1, 2])

    def test_all_question_types_include_heading_rubric_and_slot(self):
        ns = api_namespace(self.model)
        req = ns["JevRequest"](state="Shared state", questions={
            "yes": {"type": "noul", "instructions": "Boolean heading?"},
            "score": {"type": "score", "instructions": "Score heading?", "criteria": ["low", "high"]},
            "choice": {"type": "choice", "instructions": "Choice heading?", "criteria": {"a": "first", "b": "second"}},
        })
        prompt, slots, _ = ns["build_jev"](req, first_question_number=4)
        ids, positions, groups = ns["build_masked_input"](prompt, slots)
        allowed = build_slot_attention_mask(groups)[0, 0]
        for i, ((start, end, group), slot, position) in enumerate(zip(prompt.answer_spans, slots, positions)):
            self.assertTrue(prompt.text[start:end].lstrip().startswith(f"Question {i + 4}"))
            self.assertEqual(slot.insertion_offset, end)
            self.assertEqual(group, i + 4)
            self.assertEqual(groups[position].item(), group)
            self.assertEqual(ids[0, position].item(), 255)
            own = (groups == group).nonzero().flatten()
            self.assertEqual(own[-1].item(), position)
            self.assertTrue(allowed[position, own].all())
            self.assertTrue(allowed[own, position].all())
            for other in positions:
                self.assertEqual(allowed[position, other].item(), position == other)

    def test_grouped_answer_assembly_and_request_override_ignored(self):
        ns = api_namespace(self.model)
        questions = {
            "a": {"type":"noul", "instructions":"A?"},
            "b": {"type":"choice", "instructions":"B?", "criteria":{"x":"yes", "y":"no", "z":"maybe"}},
            "c": {"type":"score", "instructions":"C?", "criteria":["low","medium","high"]},
            "d": {"type":"choice", "instructions":"D?", "criteria":{f"o{i}":str(i) for i in range(27)}},
        }
        def evaluate(prompt, slots):
            ids, positions, groups = ns["build_masked_input"](prompt, slots)
            candidates = [list(range(10, 10 + len(slot.allowed))) for slot in slots]
            result = decode(self.model, ids, positions, candidates, mask_id=255, group_ids=groups)
            return ([s.allowed[i] for s, i in zip(slots, result.winners)], result.logits,
                    result.probabilities, ids.shape[1], result.forward_passes)
        ns["evaluate"] = evaluate
        for limit in (1, 2, 3, 128):
            ns["MAX_QUESTIONS_PER_PROMPT"] = limit
            for flag in (None, False, True):
                req = ns["JevRequest"](state="Hanna Jeff", questions=questions, isolated=flag)
                groups = ns["build_jev_groups"](req)
                ns["inference_lock"] = asyncio.Lock()
                answers, tokens, passes = asyncio.run(ns["evaluate_jev_groups"](groups))
                self.assertEqual(list(answers), list(questions))
                self.assertEqual(passes, len(groups))
                self.assertEqual(len(answers["d"]["probabilities"]), 27)
                self.assertGreater(tokens, 0)


if __name__ == "__main__":
    unittest.main()
