# SPDX-License-Identifier: Apache-2.0
"""Exercise decoder scheduling and cache invariants with controlled logits."""

import unittest
from types import SimpleNamespace

import torch

from maskdecide.decoding import decode


MASK = 99


class FakeModel:
    """A shifted-logit model whose clean prefix cache can be inspected."""

    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def __call__(self, input_ids, past_key_values=None, use_cache=False,
                 update_past_key_values=False, block_size=32, logits_to_keep=0):
        prefix = [] if past_key_values is None else past_key_values.tokens
        incoming = input_ids[0].tolist()
        sequence = prefix + incoming
        self.calls.append((list(prefix), list(incoming), update_past_key_values))
        if update_past_key_values:
            if MASK in sequence:
                raise AssertionError("An unresolved mask was committed to the KV cache")
            cache = SimpleNamespace(tokens=list(sequence))
        else:
            cache = past_key_values
        logits = torch.zeros((1, len(incoming), 100))
        for i in range(len(incoming)):
            # Make an invalid token the full-vocabulary winner to check that
            # the decoder always restricts its selections to allowed tokens.
            logits[0, i, 77] = 1000
            logits[0, i, 1:3] = torch.tensor(self.scores(len(prefix) + i, sequence))
        selection = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        return SimpleNamespace(logits=logits[:, selection], past_key_values=cache)


def masked_input(length, positions):
    tokens = torch.full((1, length), 5, dtype=torch.long)
    tokens[0, positions] = MASK
    return tokens


class DecoderTests(unittest.TestCase):
    def test_high_probability_slot_is_filled_first_then_other_slot_is_recomputed(self):
        def scores(pos, sequence):
            if pos == 34:
                return [0.0, 10.0]
            if pos == 33:
                return [0.0, 4.0] if sequence[35] == 2 else [1.0, 0.0]
            return [0.0, 0.0]

        model = FakeModel(scores)
        tokens = masked_input(36, [34, 35])
        result = decode(model, tokens, [34, 35], [[1, 2], [1, 2]], mask_id=MASK)
        self.assertEqual(result.winners, [1, 1])
        self.assertEqual(result.logits, [[0.0, 4.0], [0.0, 10.0]])
        self.assertEqual(result.forward_passes, 3)  # prefill + two rounds
        self.assertEqual(model.calls[2][1], [5, 5, MASK, 2])
        self.assertEqual(tokens[0, [34, 35]].tolist(), [MASK, MASK])  # caller input is immutable

    def test_block_boundary_uses_previous_block_logits(self):
        model = FakeModel(lambda pos, seq: [0.0, 8.0] if pos == 31 else [8.0, 0.0])
        result = decode(model, masked_input(65, [32, 64]), [32, 64], [[1, 2]] * 2, mask_id=MASK)
        self.assertEqual(result.winners, [1, 0])
        self.assertEqual(result.forward_passes, 2)
        self.assertTrue(all(update for _, _, update in model.calls))
        self.assertEqual(model.calls[1][1][0], 2)  # resolved boundary token enters cache

    def test_masks_on_both_sides_of_boundary_do_not_cache_stale_tokens(self):
        def scores(pos, sequence):
            if pos == 30 or (pos == 31 and sequence[31] == 2):
                return [0.0, 8.0]
            return [8.0, 0.0]

        model = FakeModel(scores)
        result = decode(model, masked_input(33, [31, 32]), [31, 32], [[1, 2]] * 2, mask_id=MASK)
        self.assertEqual(result.winners, [1, 1])
        self.assertEqual(model.calls[0][1][-1], MASK)
        self.assertEqual(model.calls[1][1][-1], 2)
        self.assertTrue(model.calls[1][2])

    def test_tied_scores_make_progress_with_a_bounded_number_of_calls(self):
        model = FakeModel(lambda pos, seq: [0.0, 0.0])
        result = decode(model, masked_input(8, [2, 4, 6]), [2, 4, 6], [[1, 2]] * 3,
                        mask_id=MASK, threshold=1.0)
        self.assertEqual(result.winners, [0, 0, 0])
        self.assertEqual(result.forward_passes, 3)
        self.assertEqual(model.calls[1][1][2], 1)
        self.assertEqual(model.calls[2][1][4], 1)

    def test_strong_slots_in_same_block_can_be_filled_together(self):
        model = FakeModel(lambda pos, seq: [9.0, 0.0])
        result = decode(model, masked_input(8, [2, 4, 6]), [2, 4, 6], [[1, 2]] * 3, mask_id=MASK)
        self.assertEqual(result.winners, [0, 0, 0])
        self.assertEqual(result.forward_passes, 1)

    def test_one_pass_keeps_original_shift_and_candidate_selection(self):
        model = FakeModel(lambda pos, seq: [9.0, 0.0] if pos == 2 else [0.0, 9.0])
        result = decode(model, masked_input(6, [3, 5]), [3, 5], [[1, 2]] * 2,
                        mask_id=MASK, mode="one_pass")
        self.assertEqual(result.winners, [0, 1])
        self.assertEqual(result.forward_passes, 1)
        self.assertFalse(model.calls[0][2])

    def test_nonfinite_scores_fail_instead_of_silently_selecting_an_answer(self):
        for value in [float("nan"), float("inf"), float("-inf")]:
            with self.subTest(value=value):
                model = FakeModel(lambda pos, seq: [value, 0.0])
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    decode(model, masked_input(4, [2]), [2], [[1, 2]], mask_id=MASK)

    def test_unregistered_masks_are_rejected_before_prefill(self):
        model = FakeModel(lambda pos, seq: [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "registered"):
            decode(model, masked_input(40, [2, 35]), [35], [[1, 2]], mask_id=MASK)
        self.assertEqual(model.calls, [])

    def test_invalid_configuration_is_rejected(self):
        for settings in [{"threshold": 0}, {"threshold": 1.1}, {"threshold": float("nan")}, {"mode": "unknown"}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                decode(FakeModel(lambda pos, seq: [0.0, 0.0]), masked_input(4, [2]), [2], [[1, 2]],
                       mask_id=MASK, **settings)


if __name__ == "__main__":
    unittest.main()
