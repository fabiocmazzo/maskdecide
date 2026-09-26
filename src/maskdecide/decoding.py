# SPDX-License-Identifier: Apache-2.0
"""Constrained Nemotron decoding with optional per-question attention."""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F

from maskdecide.slot_attention import build_slot_attention_mask, slot_attention_scope


DecoderMode = Literal["one_pass", "iterative"]


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
    mode: DecoderMode = "one_pass",
    threshold: float = 0.9,
    group_ids: torch.Tensor | None = None,
) -> DecodeResult:
    """Fill registered masks; capture each slot's logits before commitment.

    With group_ids, all tokens use shared-state/per-question attention in
    every forward. Group membership does not change when masks are filled.
    API builders assign a positive group to every question, including scores.
    Group 0 is shared context; low-level callers may also place slots there.
    Without group_ids, preserve the original unrestricted diffusion path.
    """
    if mode not in {"one_pass", "iterative"}:
        raise ValueError("Invalid decoder mode")
    if not 0 < threshold <= 1:
        raise ValueError("Invalid decoder threshold")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("Expected a single input sequence")
    if not positions or positions != sorted(set(positions)):
        raise ValueError("Answer positions must be nonempty, unique and sorted")
    if len(positions) != len(candidates):
        raise ValueError("Candidates must match answer positions")
    if any(not ids or mask_id in ids or len(set(ids)) != len(ids) for ids in candidates):
        raise ValueError("Invalid or duplicate candidate tokens")
    if positions[0] < 0 or positions[-1] >= input_ids.shape[1]:
        raise ValueError("Answer position outside input sequence")
    actual_masks = (input_ids[0] == mask_id).nonzero().flatten().tolist()
    if actual_masks != positions:
        raise ValueError("Input masks must match registered answer positions")

    # Build visibility once: replacing a MASK with a label changes token values,
    # never sequence length, position, or attention-group membership.
    allowed = None
    slot_groups = None
    if group_ids is not None:
        if group_ids.shape != (input_ids.shape[1],) or group_ids.device != input_ids.device:
            raise ValueError("group_ids must match input length and device")
        allowed = build_slot_attention_mask(group_ids)
        slot_groups = group_ids[positions].tolist()
        if any(group < 0 for group in slot_groups):
            raise ValueError("Answer slots must not use negative groups")

    vocab_size = model.diffusion_head.weight.shape[0]
    for i, ids in enumerate(candidates):
        if any(not 0 <= token_id < vocab_size for token_id in ids):
            raise ValueError(f"Invalid candidate token ID for slot {i}; vocabulary={vocab_size}")

    winners = [-1] * len(positions)
    raw = [[] for _ in positions]
    probs = [[] for _ in positions]
    tokens = input_ids.clone()
    calls = 0

    def select(values, index):
        restricted = values.float()
        if restricted.numel() != len(candidates[index]):
            raise ValueError(f"Candidate logit count mismatch for slot {index}")
        if not torch.isfinite(restricted).all().item():
            raise ValueError("Non-finite candidate logits")
        # Normalize only over this slot's permitted labels, not the vocabulary.
        # This preference drives scheduling; it is not calibrated certainty.
        probabilities = restricted.softmax(dim=-1)
        winner = int(restricted.argmax().item())
        confidence = float(probabilities[winner].item())
        return winner, restricted, probabilities, confidence

    def commit(index, winner, values, probabilities):
        # Save scores computed while this slot was still masked. Rescoring after
        # filling it could let bidirectional attention reveal its own answer.
        winners[index] = winner
        raw[index] = values.tolist()
        probs[index] = probabilities.tolist()
        tokens[0, positions[index]] = candidates[index][winner]

    def predict(indices):
        nonlocal calls
        scope = slot_attention_scope(model, allowed) if allowed is not None else nullcontext()
        # Scope reaches the registered attention backend even when upstream
        # diffusion layers replace the ordinary attention_mask with None.
        with scope:
            output = model(
                input_ids=tokens,
                # Recompute the full bidirectional sequence after commitments;
                # cached states could still reflect the old unresolved masks.
                use_cache=False,
                use_causal_mask=False,
                output_last_hidden_states_only=True,
            )
        calls += 1
        selected_positions = torch.tensor(
            [positions[i] for i in indices], dtype=torch.long, device=tokens.device,
        )
        # Nemotron predicts at the MASK position itself (no preceding-token
        # shift). Read only the slots currently being evaluated.
        hidden = output.last_hidden_state[0, selected_positions, :]
        # Project onto the union of candidate vocabulary rows instead of
        # allocating full-vocabulary logits for every sequence position.
        allowed_ids = sorted({token_id for i in indices for token_id in candidates[i]})
        token_indices = torch.tensor(allowed_ids, dtype=torch.long, device=hidden.device)
        selected_weights = model.diffusion_head.weight.index_select(0, token_indices)
        bias = getattr(model.diffusion_head, "bias", None)
        selected_bias = None if bias is None else bias.index_select(0, token_indices)
        # The model runs in bfloat16, but close A/B scores can round to a tie
        # if the projection is computed there. Compare candidates in float32.
        restricted_logits = F.linear(
            hidden.float(), selected_weights.float(),
            None if selected_bias is None else selected_bias.float(),
        )
        id_to_column = {token_id: column for column, token_id in enumerate(allowed_ids)}
        return [
            restricted_logits[row, [id_to_column[token_id] for token_id in candidates[i]]]
            for row, i in enumerate(indices)
        ]

    # One pass scores every slot against the same still-masked input.
    if mode == "one_pass":
        indices = list(range(len(positions)))
        logits = predict(indices)
        for row_index, i in enumerate(indices):
            winner, values, probabilities, _ = select(logits[row_index], i)
            commit(i, winner, values, probabilities)
        return DecodeResult(winners, raw, probs, calls)

    # Iteration exposes committed labels to the next forward and reevaluates
    # only unresolved answers. At least one slot per active group must progress.
    pending = list(range(len(positions)))
    while pending:
        logits = predict(pending)
        proposals = []
        for row_index, i in enumerate(pending):
            winner, values, probabilities, confidence = select(logits[row_index], i)
            proposals.append((i, winner, values, probabilities, confidence))

        if slot_groups is None:
            # Preserve the original global scheduler when attention is disabled.
            accepted = [proposal for proposal in proposals if proposal[4] >= threshold]
            if not accepted:
                accepted = [max(proposals, key=lambda item: item[4])]
        else:
            # Scheduling is per group: each question makes
            # progress even when a different question has confident answers.
            # Any low-level group-0 slots share one scheduling bucket. This
            # scheduling rule does not imply isolation through shared context.
            by_group = {}
            for proposal in proposals:
                by_group.setdefault(slot_groups[proposal[0]], []).append(proposal)
            accepted = []
            for group_proposals in by_group.values():
                confident = [p for p in group_proposals if p[4] >= threshold]
                accepted.extend(confident or [max(group_proposals, key=lambda item: item[4])])

        completed = set()
        for i, winner, values, probabilities, _ in accepted:
            commit(i, winner, values, probabilities)
            completed.add(i)
        pending = [i for i in pending if i not in completed]

    return DecodeResult(winners, raw, probs, calls)
