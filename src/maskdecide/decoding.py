# SPDX-License-Identifier: Apache-2.0
"""Constrained decoding for Fast-dLLM v2's shifted, block-causal logits."""

from dataclasses import dataclass
from typing import Literal

import torch


DecoderMode = Literal["one_pass", "iterative"]
BLOCK_SIZE = 32


@dataclass
class DecodeResult:
    winners: list[int]
    logits: list[list[float]]
    probabilities: list[list[float]]
    forward_passes: int


@torch.inference_mode()
def decode(
    model,
    input_ids: torch.Tensor,
    positions: list[int],
    candidates: list[list[int]],
    *,
    mask_id: int,
    mode: DecoderMode = "iterative",
    threshold: float = 0.9,
) -> DecodeResult:
    """Fill only registered answer slots, preserving all scaffold tokens.

    Iterative mode caches complete, clean blocks only. Within the active block,
    commit predictions whose restricted softmax exceeds the scheduling threshold,
    or the strongest remaining prediction if none does. Every round makes
    progress, so there are at most two model calls per slot (including prefills).

    Return each slot's logits from BEFORE committing its answer. Rescoring filled
    slots would leak their own answer through bidirectional block attention.
    Scheduling probabilities are not calibrated answer confidence.
    """
    if mode not in {"one_pass", "iterative"}:
        raise ValueError("Decoder mode must be one_pass or iterative")
    if not 0 < threshold <= 1:
        raise ValueError("Decoder threshold must be in (0, 1]")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Decoder requires a single input sequence")
    if not positions or positions != sorted(set(positions)):
        raise ValueError("Answer positions must be nonempty, unique, and increasing")
    if len(positions) != len(candidates) or any(not ids or mask_id in ids for ids in candidates):
        raise ValueError("Each answer position requires non-mask candidate tokens")
    if positions[0] < 1 or positions[-1] >= input_ids.shape[1]:
        raise ValueError("Answer positions must have a preceding token and lie within the input")
    actual_masks = (input_ids[0] == mask_id).nonzero().flatten().tolist()
    if actual_masks != positions:
        raise ValueError("Input masks must match the registered answer positions")

    winners = [-1] * len(positions)
    raw: list[list[float]] = [[] for _ in positions]
    probs: list[list[float]] = [[] for _ in positions]
    calls = 0

    def select(row, index):
        values = row[candidates[index]].float()
        if not torch.isfinite(values).all().item():
            raise ValueError("Model returned non-finite candidate logits")
        winner = int(values.argmax().item())
        return winner, values

    if mode == "one_pass":
        # Preserve the original forward shape and selection for comparisons.
        output = model(input_ids=input_ids, use_cache=False, block_size=BLOCK_SIZE)
        for i, pos in enumerate(positions):
            winner, values = select(output.logits[0, pos - 1], i)
            winners[i], raw[i] = winner, values.tolist()
            probs[i] = values.softmax(dim=-1).tolist()
        return DecodeResult(winners, raw, probs, 1)

    tokens = input_ids.clone()
    cache = None
    cached_length = 0
    groups: dict[int, list[int]] = {}
    for i, pos in enumerate(positions):
        groups.setdefault(pos // BLOCK_SIZE, []).append(i)

    def commit(index, winner, values):
        winners[index] = winner
        raw[index] = values.tolist()
        probs[index] = values.softmax(dim=-1).tolist()
        tokens[0, positions[index]] = candidates[index][winner]

    for block, indices in groups.items():
        start = block * BLOCK_SIZE
        end = min(start + BLOCK_SIZE, tokens.shape[1])
        prefix_last = None
        if cached_length < start:
            # All earlier answer blocks are now complete. Never cache noisy
            # representations: they would persist after their masks are filled.
            output = model(
                input_ids=tokens[:, cached_length:start],
                past_key_values=cache,
                use_cache=True,
                update_past_key_values=True,
                block_size=BLOCK_SIZE,
                logits_to_keep=1,
            )
            calls += 1
            cache = output.past_key_values
            cached_length = start
            prefix_last = output.logits[0, -1]

        pending = list(indices)
        if positions[pending[0]] == start:
            # The first token in a block is predicted by the PREVIOUS block's
            # last position. Indexing -1 in the active block would read its end.
            if prefix_last is None:
                raise RuntimeError("Missing prefix logits for a block-boundary answer")
            i = pending.pop(0)
            winner, values = select(prefix_last, i)
            commit(i, winner, values)

        while pending:
            prediction_positions = torch.tensor(
                [positions[i] - start - 1 for i in pending],
                dtype=torch.long,
                device=tokens.device,
            )
            output = model(
                input_ids=tokens[:, start:end],
                past_key_values=cache,
                use_cache=True,
                update_past_key_values=False,
                block_size=BLOCK_SIZE,
                logits_to_keep=prediction_positions,
            )
            calls += 1
            proposals = []
            for row_index, i in enumerate(pending):
                winner, values = select(output.logits[0, row_index], i)
                probability = float(values.softmax(dim=-1)[winner].item())
                proposals.append((i, winner, values, probability))
            accepted = [proposal for proposal in proposals if proposal[3] >= threshold]
            if not accepted:
                accepted = [max(proposals, key=lambda proposal: proposal[3])]
            for i, winner, values, _ in accepted:
                commit(i, winner, values)
            completed = {proposal[0] for proposal in accepted}
            pending = [i for i in pending if i not in completed]

    return DecodeResult(winners, raw, probs, calls)
