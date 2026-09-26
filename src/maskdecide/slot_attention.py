# SPDX-License-Identifier: Apache-2.0
"""Inference-only attention routing for the Nemotron Ministral3 encoder.

Groups: 0 = shared state; positive IDs = question, rubric and answer slots.
Questions attend to the shared state and their own group only; the shared
state keeps full visibility. Consequently, questions can still influence each
other indirectly through shared-state representations across encoder layers.
The mask is injected at the attention backend because published diffusion
forwards discard the ordinary attention_mask argument. No weights, projection
layers, positional embeddings or upstream source files are replaced.
"""

import copy
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


BACKEND_NAME = "maskdecide_question_sdpa"


@dataclass
class _ActiveMask:
    allowed: torch.Tensor
    layer_ids: tuple[int, ...]
    visits: dict[int, int] = field(default_factory=dict)


_active_mask: ContextVar[_ActiveMask | None] = ContextVar(
    "maskdecide_active_attention_mask", default=None,
)


def build_slot_attention_mask(group_ids: torch.Tensor) -> torch.Tensor:
    """Return boolean SDPA permissions [batch=1, heads=1, query, key].

    True means visible (PyTorch SDPA convention). The singleton head dimension
    broadcasts the same routing policy to every attention head. No padding.
    """
    if group_ids.ndim != 1 or group_ids.numel() == 0:
        raise ValueError("group_ids must be a nonempty one-dimensional tensor")
    if group_ids.dtype != torch.long:
        raise ValueError("group_ids must use torch.long")
    if (group_ids < 0).any().item():
        raise ValueError("Negative group IDs are not supported")
    if not (group_ids == 0).any().item():
        raise ValueError("At least one shared-state token is required")
    queries = group_ids[:, None]
    keys = group_ids[None, :]
    # Rows are readers (queries); columns are visible sources (keys).
    # Group 0 can read everything. A positive group can read group 0 and itself,
    # including its answer label and mask, but cannot directly read other groups.
    # This blocks direct edges only: question B -> state -> question A remains
    # possible over successive layers because shared-state rows see all groups.
    return ~((queries != 0) & (keys != 0) & (queries != keys))[None, None]


def _make_backend(original_sdpa):
    def question_sdpa(
        module, query, key, value, attention_mask=None,
        dropout=0.0, scaling=None, **kwargs,
    ):
        active = _active_mask.get()
        if active is None:
            return original_sdpa(
                module, query, key, value, attention_mask,
                dropout=dropout, scaling=scaling, **kwargs,
            )

        layer_id = id(module)
        if layer_id not in active.layer_ids:
            raise RuntimeError("Unexpected attention layer inside masked forward")
        if module.training or dropout != 0:
            raise RuntimeError("Slot attention supports eval/inference only")
        length = active.allowed.shape[-1]
        if (
            query.shape[0] != 1 or key.shape[0] != 1
            or query.shape[-2] != length or key.shape[-2] != length
            or value.shape[-2] != length
        ):
            raise RuntimeError("Slot attention requires one full sequence, without KV cache")
        if active.allowed.device != query.device:
            raise RuntimeError("Attention mask and queries must be on the same device")

        # Preserve grouped-query attention without relying on a particular
        # Transformers SDPA wrapper's GQA support or boolean-mask conversion.
        if key.shape[1] != value.shape[1] or query.shape[1] % key.shape[1]:
            raise RuntimeError("Incompatible query/key/value head counts")
        repeats = query.shape[1] // key.shape[1]
        if repeats != 1:
            key = key.repeat_interleave(repeats, dim=1)
            value = value.repeat_interleave(repeats, dim=1)

        # Apply the routing matrix at the operation that actually mixes values.
        # Diffusion is bidirectional, so no causal triangle is added here.
        output = F.scaled_dot_product_attention(
            query, key, value,
            attn_mask=active.allowed,
            dropout_p=0.0,
            is_causal=False,
            scale=scaling,
        )
        active.visits[layer_id] = active.visits.get(layer_id, 0) + 1
        # Transformers attention backends return [batch, seq, heads, dim].
        return output.transpose(1, 2).contiguous(), None

    question_sdpa._maskdecide_backend = True
    return question_sdpa


def install_slot_attention(model) -> int:
    """Route only this model's attention layers through our SDPA backend.

    Call once after loading. The encoder's configuration remains unchanged.
    Unsupported model classes fail explicitly, rather than running unmasked.
    """
    if hasattr(model, "_maskdecide_layer_ids"):
        return len(model._maskdecide_layer_ids)
    if model.training:
        raise RuntimeError("Call model.eval() before installing slot attention")
    layers = getattr(getattr(model, "encoder", None), "layers", None)
    if layers is None or len(layers) == 0:
        raise RuntimeError("Expected a Nemotron encoder with self-attention layers")

    prepared = []
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        if type(attn).__name__ != "Ministral3Attention":
            raise RuntimeError(
                f"Unsupported attention class: {type(attn).__name__}. "
                "Expected Ministral3Attention in bidirectional mode."
            )
        forward = inspect.unwrap(type(attn).forward)
        registry = getattr(forward, "__globals__", {}).get("ALL_ATTENTION_FUNCTIONS")
        if registry is None or not callable(getattr(registry, "register", None)):
            raise RuntimeError("Attention class does not expose the supported backend registry")
        original_sdpa = registry["sdpa"]
        if BACKEND_NAME in registry:
            if not getattr(registry[BACKEND_NAME], "_maskdecide_backend", False):
                raise RuntimeError("Attention backend name is already in use")
        else:
            registry.register(BACKEND_NAME, _make_backend(original_sdpa))
        # Copy configuration per attention module so dispatch changes stay local
        # to these layers instead of mutating the encoder's shared config.
        config = copy.copy(attn.config)
        config._attn_implementation = BACKEND_NAME
        prepared.append((attn, config))

    for attn, config in prepared:
        attn.config = config
        attn.diffusion_lm = True
    model._maskdecide_layer_ids = tuple(id(attn) for attn, _ in prepared)
    return len(prepared)


@contextmanager
def slot_attention_scope(model, allowed: torch.Tensor):
    """Apply a mask to one synchronous forward and verify every layer used it."""
    expected = getattr(model, "_maskdecide_layer_ids", ())
    if not expected:
        raise RuntimeError("install_slot_attention(model) must run before masked decoding")
    if _active_mask.get() is not None:
        raise RuntimeError("Nested masked forwards are not supported")
    if (
        allowed.dtype != torch.bool or allowed.ndim != 4
        or allowed.shape[:2] != (1, 1)
        or allowed.shape[-2] != allowed.shape[-1]
    ):
        raise ValueError("Expected a boolean attention mask [1, 1, L, L]")
    active = _ActiveMask(allowed, expected)
    # Context-local state carries permissions into the registered backend even
    # if upstream forward methods discard their attention_mask argument.
    handle = _active_mask.set(active)
    try:
        yield
        # Fail closed if any layer skipped the backend: a returned answer must
        # never silently bypass the requested attention routing.
        if active.visits != {layer_id: 1 for layer_id in expected}:
            raise RuntimeError(
                "Structured attention was not applied exactly once in every encoder layer. "
                "Check the loaded model implementation; output was rejected."
            )
    finally:
        _active_mask.reset(handle)


@torch.inference_mode()
def verify_model_isolation(model, token_ids: list[int], mask_id: int) -> dict:
    """Short forwards on the loaded checkpoint, not an accuracy benchmark.

    Only token values change, so sequence length and RoPE positions stay fixed.
    The probe verifies the invariants the decoder relies on, whichever mask
    direction is configured: the mask must actually reach every layer, the
    shared state must influence every answer slot, and questions must stay
    visible to the shared state. A wrong direction (shared state hidden from
    questions) fails the state-visibility check; a dropped or ignored mask
    fails the layer-visit accounting in slot_attention_scope.
    """
    if len(set(token_ids)) < 6:
        raise ValueError("Supply six distinct ordinary token IDs for the probe")
    device = model.get_input_embeddings().weight.device
    a, b, c, d, e, f = token_ids[:6]
    tokens = torch.tensor([[a, b, c, d, mask_id, e, f, mask_id]], device=device)
    groups = torch.tensor([0, 0, 1, 1, 1, 2, 2, 2], device=device)
    allowed = build_slot_attention_mask(groups)

    def hidden(ids):
        with slot_attention_scope(model, allowed):
            result = model(
                input_ids=ids, use_cache=False, use_causal_mask=False,
                output_last_hidden_states_only=True,
            )
        return result.last_hidden_state.float()

    reference = hidden(tokens)
    changed_question = tokens.clone()
    changed_question[0, 5:7] = torch.tensor([a, b], device=device)
    other = hidden(changed_question)
    if not torch.isfinite(reference).all() or not torch.isfinite(other).all():
        raise RuntimeError("Non-finite hidden states during attention probe")
    state_delta = (reference[:, :2] - other[:, :2]).abs().max().item()
    if state_delta <= 1e-6:
        raise RuntimeError(
            "Question tokens are invisible to the shared state; the mask "
            "direction hides answers from the state representations they need"
        )
    changed_state = tokens.clone()
    changed_state[0, :2] = torch.tensor([e, f], device=device)
    shared = hidden(changed_state)
    if not torch.isfinite(shared).all():
        raise RuntimeError("Non-finite hidden states during shared-state probe")
    deltas = [(reference[0, i] - shared[0, i]).abs().max().item() for i in (4, 7)]
    if any(delta <= 1e-6 for delta in deltas):
        raise RuntimeError("Shared-state probe did not affect both answer positions")
    cross_slot = (reference[0, 4] - other[0, 4]).abs().max().item()
    return {
        "layers": len(model._maskdecide_layer_ids),
        "cross_question_slot_delta": cross_slot,
        "state_visibility_delta": state_delta,
        "shared_state_slot_deltas": deltas,
    }
