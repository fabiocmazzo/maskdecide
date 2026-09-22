# SPDX-License-Identifier: Apache-2.0
"""Fast-dLLM constrained decision API + HTTP adapter for Jev System One.

Run: uv run maskdecide --host 127.0.0.1 --port 8000

This is an experimental Jev HTTP adapter, NOT the Jev model.
It returns discrete decisions: noul 0/1, choice and score with one-hot
probabilities and confidence=1.0 as a RESPONSE CONVENTION, not calibrated confidence.
"""

import json
import logging
import os
import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from maskdecide.decoding import decode

logger = logging.getLogger("uvicorn.error")
#MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_7B"
MASK_ID = 151665  # mask token ID for Fast-dLLM v2
MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
MAX_QUESTIONS = 128
MAX_CHOICE_OPTIONS = 255
MAX_LOCAL_OPTIONS = 20
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")  # optional, intended for localhost tests
DECODER_MODE = os.getenv("DECODER_MODE", "iterative")
DECODER_THRESHOLD = float(os.getenv("DECODER_THRESHOLD", "0.9"))

model = None
tokenizer = None
inference_lock = asyncio.Lock()

# Criteria keys and option IDs use the same restricted alphabet so their
# tokenization stays predictable next to the mask scaffold.
OPTION_KEY_PATTERN = r"^[a-zA-Z][a-zA-Z0-9_]*$"
OPTION_KEY_RE = re.compile(OPTION_KEY_PATTERN)


class Option(BaseModel):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    text: str = Field(min_length=1, max_length=500)


class DecisionRequest(BaseModel):
    context: str = Field(default="", max_length=20000)
    question: str = Field(min_length=1, max_length=2000)
    type: Literal["single_choice", "boolean", "multiple_choice"]
    options: list[Option] = Field(default_factory=list, max_length=MAX_LOCAL_OPTIONS)


class JevQuestion(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: Any
    criteria: Any = None


class JevRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, JevQuestion] = Field(min_length=1, max_length=MAX_QUESTIONS)
    model: str = "jev-latest"
    isolated: bool = False


@dataclass
class Slot:
    label: str  # fixed text immediately BEFORE this answer's mask
    allowed: str  # allowed one-token labels, e.g. 'AB' or 'ABCD'


@dataclass
class QuestionPlan:
    question_id: str
    kind: str
    slot_indices: list[int]
    option_keys: list[str]
    levels: list[Any]
    mode: str  # 'labels', 'binary_ranking', 'binary', 'levels'


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    if DECODER_MODE not in {"one_pass", "iterative"}:
        raise RuntimeError("DECODER_MODE must be one_pass or iterative")
    if not 0 < DECODER_THRESHOLD <= 1:
        raise RuntimeError("DECODER_THRESHOLD must be in (0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("This configuration requires a CUDA-capable GPU.")

    logger.info("Loading %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},  # tenta colocar o modelo inteiro na GPU 0
        trust_remote_code=True,
    ).eval()



    # Fail early if tokenizer differs from the expected single-token labels.
    letter_token_ids("AB")
    logger.info("Model ready | decoder=%s | threshold=%s", DECODER_MODE, DECODER_THRESHOLD)
    yield


app = FastAPI(title="MaskDecide", version="0.1.0", lifespan=lifespan)


def require_local_key(authorization: str | None) -> None:
    # No key is needed when the server is deliberately bound to 127.0.0.1.
    if LOCAL_API_KEY and authorization != f"Bearer {LOCAL_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid local API key")


def as_text(value: Any) -> str:
    """Keep structured Jev state/criteria/instructions, without dropping fields."""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_jev_request(request: JevRequest) -> None:
    if not request.model.strip():
        raise HTTPException(status_code=422, detail="model must not be empty")
    if len(as_text(request.state)) > 120_000:
        raise HTTPException(status_code=422, detail="state is too large")

    for qid, q in request.questions.items():
        if not qid or len(qid) > 200:
            raise HTTPException(status_code=422, detail="Question ID is empty or too long")
        if not isinstance(q.instructions, (str, dict, list)):
            raise HTTPException(status_code=422, detail=f"{qid}: instructions must be a string, object, or array")
        if q.type == "choice":
            if not isinstance(q.criteria, dict) or not 2 <= len(q.criteria) <= MAX_CHOICE_OPTIONS:
                raise HTTPException(status_code=422, detail=f"{qid}: choice requires 2 to 255 options")
            for key in q.criteria:
                if not isinstance(key, str) or not key:
                    raise HTTPException(status_code=422, detail=f"{qid}: Invalid option key")
                if len(key) > 40 or not OPTION_KEY_RE.fullmatch(key):
                    raise HTTPException(
                        status_code=422,
                        detail=f"{qid}: option key {key!r} must start with a letter and contain only letters, digits, or underscores (40 characters maximum)",
                    )
                # The key is interpolated immediately before a mask token in the
                # binary_ranking scaffold; it must tokenize without a mask.
                if tokenizer is not None and MASK_ID in tokenizer.encode(key, add_special_tokens=False):
                    raise HTTPException(status_code=422, detail=f"{qid}: option key {key!r} collides with the model mask token")
        elif q.type == "score":
            if not isinstance(q.criteria, list) or not 2 <= len(q.criteria) <= 10:
                raise HTTPException(status_code=422, detail=f"{qid}: score requires 2 to 10 levels")
        elif q.criteria is not None:
            if not isinstance(q.criteria, dict) or any(k not in {"true", "false"} for k in q.criteria):
                raise HTTPException(status_code=422, detail=f"{qid}: noul criteria may only contain true/false keys")


def validate_local_request(request: DecisionRequest) -> None:
    if request.type == "boolean":
        if request.options:
            raise HTTPException(status_code=422, detail="boolean does not accept options")
        return
    if len(request.options) < 2:
        raise HTTPException(status_code=422, detail="Choice requires at least 2 options")
    ids = [option.id for option in request.options]
    if len(ids) != len(set(ids)):
        raise HTTPException(status_code=422, detail="Option IDs must be unique")


def letter_token_ids(allowed: str) -> dict[str, int]:
    result = {}
    for letter in allowed:
        ids = tokenizer.encode(f" {letter}", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"Label {letter!r} does not encode as one token: {ids}")
        result[letter] = ids[0]
    return result


def format_prompt(parts: list[str]) -> str:
    messages = [{"role": "user", "content": "\n".join(parts)}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def build_local(request: DecisionRequest) -> tuple[str, list[Slot]]:
    """Preserve the working local /decide behavior and its per-option labels."""
    parts = [
        "Answer using the supplied context. Choose only from the allowed labels.",
        "Do not explain your answer.",
    ]
    if request.context:
        parts.append(f"\nContext:\n{request.context}")
    parts.append(f"\nQuestion:\n{request.question}")

    if request.type == "single_choice":
        parts.append("\nChoose exactly ONE option:")
        parts.extend(f"{LETTERS[i]}: {option.text}" for i, option in enumerate(request.options))
        parts.append("\nReturn only the letter of the selected option.")
        slots = [Slot("\nAnswer:", LETTERS[:len(request.options)])]
    elif request.type == "boolean":
        parts.extend(["\nAnswer the statement as true or false.", "A: True", "B: False"])
        slots = [Slot("\nAnswer:", "AB")]
    else:
        parts.append("\nFor EACH option below, decide independently whether it should be selected.")
        parts.extend(f"Option {i}: {option.text}" for i, option in enumerate(request.options, 1))
        parts.extend([
            "\nFor each option, answer:",
            "A: Yes, select this option.",
            "B: No, do not select this option.",
            "Answer every option. Multiple options may be selected.",
        ])
        slots = [Slot(f"\nThe option {i} must be selected?", "AB")
                 for i in range(1, len(request.options) + 1)]
    return format_prompt(parts), slots


def build_jev(request: JevRequest) -> tuple[str, list[Slot], list[QuestionPlan]]:
    """Build a shared prompt and constrained answer slots for all questions."""
    parts = [
        "Evaluate the shared state. Answer each numbered question from its own rubric.",
        "Output fields are fixed. Select only the permitted labels; no explanations.",
        f"\nShared state:\n{as_text(request.state)}",
    ]
    slots: list[Slot] = []
    plans: list[QuestionPlan] = []

    for number, (qid, q) in enumerate(request.questions.items(), start=1):
        parts.append(f"\nQuestion {number} ({q.type}): {as_text(q.instructions)}")
        first_slot = len(slots)
        if q.type == "noul":
            parts.extend(["A: Yes / True", "B: No / False"])
            if q.criteria:
                for truth in ("true", "false"):
                    if truth in q.criteria:
                        parts.append(f"{truth}: {as_text(q.criteria[truth])}")
            slots.append(Slot(f"\nQuestion {number}: Is the answer yes/true? Answer:", "AB"))
            plans.append(QuestionPlan(qid, "noul", [first_slot], [], [], "binary"))
        elif q.type == "score":
            levels = list(q.criteria)
            parts.append("Choose the level which best matches the state:")
            for i, level in enumerate(levels):
                parts.append(f"{LETTERS[i]}: {as_text(level)}")
            slots.append(Slot(f"\nQuestion {number}: Best matching level? Answer:", LETTERS[:len(levels)]))
            plans.append(QuestionPlan(qid, "score", [first_slot], [], levels, "levels"))
        else:
            criteria = q.criteria
            keys = list(criteria)
            if len(keys) <= len(LETTERS):
                parts.append("Choose exactly ONE alternative:")
                for i, key in enumerate(keys):
                    description = criteria[key]
                    detail = key if description is None else f"{key}: {as_text(description)}"
                    parts.append(f"{LETTERS[i]}: {detail}")
                slots.append(Slot(f"\nQuestion {number}: Selected option? Answer:", LETTERS[:len(keys)]))
                plans.append(QuestionPlan(qid, "choice", [first_slot], keys, [], "labels"))
            else:
                # A single alphabet token cannot encode 27..255 options.
                # For large Choices, score each option as yes/no on its own
                # mask, then rank by the raw yes/no logit margins.
                parts.append("Choose the single best option from this list:")
                for i, key in enumerate(keys, start=1):
                    description = criteria[key]
                    detail = key if description is None else f"{key}: {as_text(description)}"
                    parts.append(f"Option {i}: {detail}")
                parts.extend(["For every option, A means best option and B means not best option."])
                indices = []
                for i, key in enumerate(keys, start=1):
                    indices.append(len(slots))
                    slots.append(Slot(f"\nQuestion {number}: Should option {i} ({key}) be the selected best option? Answer:", "AB"))
                plans.append(QuestionPlan(qid, "choice", indices, keys, [], "binary_ranking"))

    return format_prompt(parts), slots, plans


def build_jev_isolated(request: JevRequest) -> list[tuple[str, list[Slot], QuestionPlan]]:
    """Build one prompt per question, with the same state and rubric.

    Isolation prevents questions from attending to each other's answer slots,
    which removes order-dependent interactions at the cost of more tokens and
    latency. Each prompt is evaluated independently; plans are returned in the
    original question order.
    """
    state_text = as_text(request.state)
    isolated: list[tuple[str, list[Slot], QuestionPlan]] = []
    for number, (qid, q) in enumerate(request.questions.items(), start=1):
        parts = [
            "Evaluate the shared state. Answer the numbered question from its rubric.",
            "Output fields are fixed. Select only the permitted labels; no explanations.",
            f"\nShared state:\n{state_text}",
            f"\nQuestion {number} ({q.type}): {as_text(q.instructions)}",
        ]
        slots: list[Slot] = []
        if q.type == "noul":
            parts.extend(["A: Yes / True", "B: No / False"])
            if q.criteria:
                for truth in ("true", "false"):
                    if truth in q.criteria:
                        parts.append(f"{truth}: {as_text(q.criteria[truth])}")
            slots.append(Slot(f"\nQuestion {number}: Is the answer yes/true? Answer:", "AB"))
            plan = QuestionPlan(qid, "noul", [0], [], [], "binary")
        elif q.type == "score":
            levels = list(q.criteria)
            parts.append("Choose the level which best matches the state:")
            for i, level in enumerate(levels):
                parts.append(f"{LETTERS[i]}: {as_text(level)}")
            slots.append(Slot(f"\nQuestion {number}: Best matching level? Answer:", LETTERS[:len(levels)]))
            plan = QuestionPlan(qid, "score", [0], [], levels, "levels")
        else:
            criteria = q.criteria
            keys = list(criteria)
            if len(keys) <= len(LETTERS):
                parts.append("Choose exactly ONE alternative:")
                for i, key in enumerate(keys):
                    description = criteria[key]
                    detail = key if description is None else f"{key}: {as_text(description)}"
                    parts.append(f"{LETTERS[i]}: {detail}")
                slots.append(Slot(f"\nQuestion {number}: Selected option? Answer:", LETTERS[:len(keys)]))
                plan = QuestionPlan(qid, "choice", [0], keys, [], "labels")
            else:
                parts.append("Choose the single best option from this list:")
                for i, key in enumerate(keys, start=1):
                    description = criteria[key]
                    detail = key if description is None else f"{key}: {as_text(description)}"
                    parts.append(f"Option {i}: {detail}")
                parts.extend(["For every option, A means best option and B means not best option."])
                indices = []
                for i, key in enumerate(keys, start=1):
                    indices.append(len(slots))
                    slots.append(Slot(f"\nQuestion {number}: Should option {i} ({key}) be the selected best option? Answer:", "AB"))
                plan = QuestionPlan(qid, "choice", indices, keys, [], "binary_ranking")
        isolated.append((format_prompt(parts), slots, plan))
    return isolated


def build_masked_input(prompt: str, slots: list[Slot]) -> tuple[torch.Tensor, list[int]]:
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    if MASK_ID in ids:
        raise HTTPException(status_code=422, detail="Input text must not contain the model mask token")
    positions = []
    for slot in slots:
        label_ids = tokenizer.encode(slot.label, add_special_tokens=False)
        if MASK_ID in label_ids:
            raise HTTPException(status_code=422, detail="Input text must not contain the model mask token")
        ids.extend(label_ids)
        positions.append(len(ids))
        ids.append(MASK_ID)
    if len(ids) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"Input with {len(ids)} tokens exceeds MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}",
        )
    return torch.tensor([ids], dtype=torch.long, device=model.device), positions


def evaluate(prompt: str, slots: list[Slot]) -> tuple[list[str], list[list[float]], int]:
    """Decode allowed labels and retain pre-commit logits for large choices."""
    input_ids, positions = build_masked_input(prompt, slots)
    token_ids = letter_token_ids("".join(dict.fromkeys("".join(s.allowed for s in slots))))
    result = decode(
        model, input_ids, positions,
        [[token_ids[letter] for letter in slot.allowed] for slot in slots],
        mask_id=MASK_ID, mode=DECODER_MODE, threshold=DECODER_THRESHOLD,
    )
    labels = [slot.allowed[winner] for slot, winner in zip(slots, result.winners, strict=True)]
    return labels, result.logits, result.probabilities, input_ids.shape[1], result.forward_passes


async def evaluate_isolated(
    isolated_prompts: list[tuple[str, list[Slot], QuestionPlan]],
) -> tuple[list[str], list[list[float]], list[list[float]], int, int]:
    """Evaluate each isolated prompt independently under the inference lock.

    The prompts share the same state but contain no other questions, so their
    answer slots cannot interact. Decoding is sequential for now; true batched
    decoding would require grouping slots across sequences.
    """
    prepared = []
    for prompt, slots, plan in isolated_prompts:
        input_ids, positions = build_masked_input(prompt, slots)
        token_ids = letter_token_ids("".join(dict.fromkeys("".join(s.allowed for s in slots))))
        prepared.append((input_ids, positions, token_ids, slots))

    all_labels: list[str] = []
    all_raw: list[list[float]] = []
    all_probs: list[list[float]] = []
    total_tokens = 0
    total_passes = 0
    async with inference_lock:
        for ids, positions, token_ids, slots in prepared:
            candidates = [[token_ids[letter] for letter in slot.allowed] for slot in slots]
            result = decode(model, ids, positions, candidates, mask_id=MASK_ID,
                            mode=DECODER_MODE, threshold=DECODER_THRESHOLD)
            total_tokens += ids.shape[1]
            total_passes += result.forward_passes
            labels = [slot.allowed[winner] for slot, winner in zip(slots, result.winners, strict=True)]
            all_labels.extend(labels)
            all_raw.extend(result.logits)
            all_probs.extend(result.probabilities)
    return all_labels, all_raw, all_probs, total_tokens, total_passes


def create_jev_answers(
    plans: list[QuestionPlan],
    labels: list[str],
    raw: list[list[float]],
    probs: list[list[float]],
) -> dict[str, dict]:
    """Translate one winning label into each Jev response shape.

    Probabilities are the restricted softmax over each slot's allowed labels,
    computed before that slot's answer was committed. They describe the model's
    preference among the permitted labels, not calibrated factual confidence.
    """
    answers: dict[str, dict] = {}
    for plan in plans:
        if plan.kind == "noul":
            slot = plan.slot_indices[0]
            selected = labels[slot]
            yes_probability = probs[slot][0] if probs[slot] else (1.0 if selected == "A" else 0.0)
            answers[plan.question_id] = {
                "type": "noul",
                "noul": yes_probability,
            }
        elif plan.kind == "score":
            slot = plan.slot_indices[0]
            selected = labels[slot]
            winner = LETTERS.index(selected)
            distribution = probs[slot] if probs[slot] else [1.0 if i == winner else 0.0 for i in range(len(plan.levels))]
            weighted_score = sum(i * p for i, p in enumerate(distribution))
            answers[plan.question_id] = {
                "type": "score",
                "score": weighted_score,
                "confidence": max(distribution),
                "legend": {str(i): level for i, level in enumerate(plan.levels)},
                "probabilities": {str(i): p for i, p in enumerate(distribution)},
            }
        elif plan.kind == "choice":
            if plan.mode == "labels":
                slot = plan.slot_indices[0]
                selected = labels[slot]
                winner = LETTERS.index(selected)
                distribution = probs[slot] if probs[slot] else [1.0 if i == winner else 0.0 for i in range(len(plan.option_keys))]
            elif plan.mode == "binary_ranking":
                # 27..255 options use an independent A/Yes vs B/No mask each.
                # No single mask already encodes the winner in this special case.
                # Compare margins ONLY to choose a single option, no softmax.
                margins = [
                    raw[index][0] - raw[index][1]
                    for index in plan.slot_indices
                ]
                winner = max(range(len(margins)), key=lambda i: margins[i])
                # Normalize the yes-probabilities of each independent slot so
                # they sum to 1. This is a heuristic response distribution,
                # not a joint categorical model over the options.
                yes_probs = [probs[index][0] if probs[index] else 0.0 for index in plan.slot_indices]
                total = sum(yes_probs)
                distribution = [p / total if total > 0 else 1.0 / len(yes_probs) for p in yes_probs]
            else:
                raise ValueError(f"Unknown choice mode: {plan.mode}")
            answers[plan.question_id] = {
                "type": "choice",
                "choice": plan.option_keys[winner],
                "confidence": max(distribution),
                "probabilities": {key: p for key, p in zip(plan.option_keys, distribution, strict=True)},
            }
        else:
            raise ValueError(f"Unknown Jev type: {plan.kind}")
    return answers


@app.post("/v1/systemone")
async def system_one(request: JevRequest, authorization: str | None = Header(default=None)):
    require_local_key(authorization)
    validate_jev_request(request)
    if model is None:
        raise HTTPException(status_code=503, detail="Model has not loaded yet")
    try:
        if request.isolated:
            isolated = build_jev_isolated(request)
            labels, raw, probs, n_tokens, passes = await evaluate_isolated(isolated)
            plans = [plan for _, _, plan in isolated]
        else:
            prompt, slots, plans = build_jev(request)
            async with inference_lock:
                labels, raw, probs, n_tokens, passes = evaluate(prompt, slots)
        answers = create_jev_answers(plans, labels, raw, probs)
    except HTTPException:
        raise
    except Exception:
        logger.exception("Internal error in the Jev adapter")
        raise HTTPException(status_code=500, detail="Internal inference failure") from None
    return {
        "model": MODEL_ID,  # NEVER report jev-latest: the backend is Fast-dLLM
        "answers": answers,
        "usage": {"input_tokens": n_tokens, "output_tokens": 0, "forward_passes": passes},
    }


@app.post("/decide")
async def decide(request: DecisionRequest, authorization: str | None = Header(default=None)):
    require_local_key(authorization)
    validate_local_request(request)
    if model is None:
        raise HTTPException(status_code=503, detail="Model has not loaded yet")
    prompt, slots = build_local(request)
    async with inference_lock:
        labels, _, _, _, _ = evaluate(prompt, slots)
    if request.type == "single_choice":
        answer: str | bool | list[str] = request.options[ord(labels[0]) - ord("A")].id
    elif request.type == "boolean":
        answer = labels[0] == "A"
    else:
        answer = [option.id for option, label in zip(request.options, labels, strict=True) if label == "A"]
    return {"type": request.type, "answer": answer, "model": MODEL_ID}


@app.get("/health")
def health():
    return {"status": "ready" if model is not None else "loading", "model": MODEL_ID}
