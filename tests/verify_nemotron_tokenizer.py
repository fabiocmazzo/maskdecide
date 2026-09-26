"""Offline layout regression with the cached real tokenizer; no model weights."""

from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

from test_jev_api_smoke import CASES
from test_slot_attention import api_namespace
from maskdecide.slot_attention import build_slot_attention_mask


def main():
    # Input assembly only needs the destination device, not a loaded model.
    device_stub = SimpleNamespace(
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(0)),
    )
    ns = api_namespace(device_stub)
    ns["tokenizer"] = AutoTokenizer.from_pretrained(
        "nvidia/Nemotron-Labs-Diffusion-3B",
        revision="0d51902da1f8869f83413ce642fab402fa5641e0",
        trust_remote_code=True,
        local_files_only=True,
    )
    ns["MASK_ID"] = 100
    checked = 0

    def verify(prompt, slots):
        nonlocal checked
        ids, positions, groups = ns["build_masked_input"](prompt, slots)
        rendered = ns["tokenizer"].decode(ids[0])
        assistant_marker = "<|im_start|>assistant"
        # Every actual answer mask belongs to the open assistant response.
        assert rendered.count(assistant_marker) == 1
        user_text, assistant_text = rendered.split(assistant_marker)
        mask_text = ns["tokenizer"].decode([ns["MASK_ID"]])
        assert mask_text not in user_text
        assert assistant_text.count(mask_text) == len(slots)
        assert "<|im_end|>" not in assistant_text
        assert positions[-1] == ids.shape[1] - 1
        allowed = build_slot_attention_mask(groups)[0, 0]
        question_count = len(prompt.answer_spans)
        for user_span, assistant_span in zip(
            prompt.question_spans[:question_count], prompt.answer_spans, strict=True,
        ):
            left, right, group = user_span
            a_left, a_right, a_group = assistant_span
            assert group == a_group
            assert prompt.text[left:right] == prompt.text[a_left:a_right]
        for slot, position in zip(slots, positions, strict=True):
            assert groups[position].item() == slot.group_id
            own = groups == slot.group_id
            other = (groups != 0) & ~own
            assert allowed[position, own].all() and allowed[own, position].all()
            assert not allowed[position, other].any()
            assert not allowed[other, position].any()
            # Both copies are private to this question, while shared state is visible.
            assert allowed[position, groups == 0].all()
            for label in slot.allowed:
                assert len(ns["tokenizer"].encode(" " + label, add_special_tokens=False)) == 1
            checked += 1
        ns["USE_SLOT_ATTENTION"] = False
        plain_ids, plain_positions, plain_groups = ns["build_masked_input"](prompt, slots)
        ns["USE_SLOT_ATTENTION"] = True
        torch.testing.assert_close(ids, plain_ids)
        assert positions == plain_positions and plain_groups is None

    for case in CASES:
        request = ns["JevRequest"](state=case["state"], questions=case["questions"])
        for prompt, slots, _ in ns["build_jev_groups"](request):
            verify(prompt, slots)
    for kind in ("boolean", "single_choice", "multiple_choice"):
        options = [] if kind == "boolean" else [
            {"id": "a", "text": "First"}, {"id": "b", "text": "Second"},
        ]
        request = ns["DecisionRequest"](
            context="Café 😀. Question 1 is quoted.", question="Select?", type=kind,
            options=options,
        )
        verify(*ns["build_local"](request))
    print(f"PASS: {len(CASES)} smoke cases + 3 local layouts; {checked} assistant slots; no weights loaded")


if __name__ == "__main__":
    main()
