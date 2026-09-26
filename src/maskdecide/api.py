# SPDX-License-Identifier: Apache-2.0
"""Fast-dLLM constrained decision API + HTTP adapter for Jev System One.

Run: uv run maskdecide --host 127.0.0.1 --port 8000

This is an experimental Jev HTTP adapter, NOT the Jev model.
Probabilities describe relative preferences among permitted labels, not
calibrated factual confidence. Jev questions are grouped exclusively by the
server-side MAX_QUESTIONS_PER_PROMPT constant; request.isolated is ignored.
"""

import asyncio
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import bitsandbytes as bnb
import torch
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field
from transformers import (
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from maskdecide.decoding import decode
from maskdecide.slot_attention import install_slot_attention, verify_model_isolation


logger = logging.getLogger("uvicorn.error")

#MASK_ID = 151665  # mask token ID for Fast-dLLM v2

MODEL_ID = "nvidia/Nemotron-Labs-Diffusion-3B"
#MODEL_ID = "nvidia/Nemotron-Labs-Diffusion-8B"

# Obtido da configuração do checkpoint no carregamento.
MASK_ID = None


MAX_INPUT_TOKENS = int(os.getenv("MAX_INPUT_TOKENS", "4096"))
MAX_QUESTIONS = 128
# Server policy: 1 = fully isolated; 3 = groups of up to three questions;
# MAX_QUESTIONS = all questions together (subject to MAX_INPUT_TOKENS).
# Request fields and JEV_ISOLATED do not override this constant.
MAX_QUESTIONS_PER_PROMPT = 30
# Both switches are server-only. Request fields never override them.
USE_SLOT_ATTENTION = True
VERIFY_SLOT_ATTENTION_ON_STARTUP = True
MAX_CHOICE_OPTIONS = 255
# Large categorical label sets strongly favor one label on the current model.
# The HTTP route compares pairs in both orders from this option count onward.
CHOICE_BINARY_MIN_OPTIONS = 15
BINARY_CHOICE_CHUNK_SIZE = 20
MAX_LOCAL_OPTIONS = 20
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def choice_labels(count: int) -> list[str]:
    """Single-token answer labels: A..Z, then two-letter pairs (AA, AB, ...).

    Only pairs that this tokenizer encodes as one token are used, so every
    label stays a single candidate token in the constrained decoder. Covers
    up to MAX_CHOICE_OPTIONS with one mask per question, replacing the old
    per-option binary ranking scaffold.
    """
    if count > MAX_CHOICE_OPTIONS:
        raise ValueError(f"Too many options: {count} > {MAX_CHOICE_OPTIONS}")
    labels = list(LETTERS)
    for pair in (first + second for first in LETTERS for second in LETTERS):
        if len(labels) >= max(count, MAX_CHOICE_OPTIONS):
            break
        # Without a loaded tokenizer (unit tests) assume pairs stay single
        # tokens; production startup validates them via letter_token_ids.
        if tokenizer is None or len(tokenizer.encode(f" {pair}", add_special_tokens=False)) == 1:
            labels.append(pair)
    if len(labels) < count:
        raise RuntimeError(
            f"Tokenizer provides only {len(labels)} single-token labels; "
            f"cannot encode {count} options on one mask"
        )
    return labels[:count]

LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")  # optional, intended for localhost tests
DECODER_MODE = os.getenv("DECODER_MODE", "one_pass")
DECODER_THRESHOLD = float(os.getenv("DECODER_THRESHOLD", "0.85"))

model = None
tokenizer = None
inference_lock = asyncio.Lock()

# Criteria keys and option IDs use the same restricted alphabet so their
# tokenization stays predictable next to the mask scaffold.
OPTION_KEY_PATTERN = r"^[a-zA-Z][a-zA-Z0-9_]*$"
OPTION_KEY_RE = re.compile(OPTION_KEY_PATTERN)


from middlewares.http_logger import HTTPLoggerMiddleware


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)



def gpu_stats(label: str):
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    peak = torch.cuda.max_memory_allocated() / 1024**3

    free, total = torch.cuda.mem_get_info()

    print(f"\n[{label}]")
    print(f"Alocado pelo PyTorch: {allocated:.2f} GiB")
    print(f"Reservado pelo PyTorch: {reserved:.2f} GiB")
    print(f"Pico alocado: {peak:.2f} GiB")
    print(f"VRAM livre na GPU: {free / 1024**3:.2f} GiB")
    print(f"VRAM total: {total / 1024**3:.2f} GiB")


def inspect_quantization(model):
    linear_4bit = sum(
        isinstance(module, bnb.nn.Linear4bit)
        for module in model.modules()
    )

    print(f"\nCamadas Linear4bit: {linear_4bit}")

    print(
        "Classe da cabeça de saída:",
        type(model.diffusion_head).__name__,
    )

    print(
        "Classe dos embeddings:",
        type(model.get_input_embeddings()).__name__,
    )

    print(
        "Tipo dos pesos da cabeça:",
        model.diffusion_head.weight.dtype,
    )


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
    isolated: bool | None = Field(
        default=None,
        description="Deprecated compatibility field. Ignored; grouping is server-controlled.",
    )


@dataclass
class Slot:
    """Describe an answer placeholder before it becomes a real mask token.

    allowed constrains output vocabulary; group_id controls attention visibility.
    These are separate constraints: valid labels do not imply correct answers.
    """

    label: str  # fixed text immediately BEFORE this answer's mask
    allowed: list[str]  # allowed one-token labels, e.g. ["A", "B"] or ["A", ..., "AA"]
    group_id: int = 1
    insertion_offset: int | None = None  # Character boundary after own question/rubric.


@dataclass
class TaggedPrompt:
    text: str
    # Half-open character intervals (start, end, group_id) in the rendered
    # chat text, before answer labels and mask tokens are inserted.
    question_spans: list[tuple[int, int, int]]
    # Only assistant-side spans receive slots; user copies share their groups.
    answer_spans: list[tuple[int, int, int]] | None = None


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
    global model, tokenizer, MASK_ID

    if DECODER_MODE not in {"one_pass", "iterative"}:
        raise RuntimeError("Invalid decoder mode")

    if not 0 < DECODER_THRESHOLD <= 1:
        raise RuntimeError("Invalid decoder threshold")

    validate_group_size()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

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

    model = AutoModel.from_pretrained(
        MODEL_ID,
        quantization_config=quantization_config,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    ).eval()

    inspect_quantization(model)

    print(
        "Attention implementation:",
        model.config._attn_implementation,
    )

    print(
        "Flash SDPA enabled:",
        torch.backends.cuda.flash_sdp_enabled(),
    )

    print(
        "Memory-efficient SDPA enabled:",
        torch.backends.cuda.mem_efficient_sdp_enabled(),
    )


    gpu_stats("Modelo carregado")

    MASK_ID = model.mask_token_id

    # Force diffusion attention instead of causal attention.
    for layer in model.encoder.layers:
        if hasattr(layer.self_attn, "diffusion_lm"):
            layer.self_attn.diffusion_lm = True

    # Validate candidate tokenization for this tokenizer.
    letter_token_ids("AB")
    if USE_SLOT_ATTENTION:
        if not tokenizer.is_fast:
            raise RuntimeError("Slot attention requires a fast tokenizer with offsets")
        layers = install_slot_attention(model)
        logger.info("SLOT ATTENTION | installed_layers=%d", layers)
        if VERIFY_SLOT_ATTENTION_ON_STARTUP:
            probe = verify_model_isolation(
                model, list(letter_token_ids("ABCDEF").values()), MASK_ID,
            )
            logger.info("SLOT ATTENTION PROBE | %s", probe)

    logger.info(
        "Model ready | decoder=%s | max_questions_per_prompt=%d | mask_id=%s",
        DECODER_MODE,
        MAX_QUESTIONS_PER_PROMPT,
        MASK_ID,
    )

    yield


app = FastAPI(title="MaskDecide", version="0.1.0", lifespan=lifespan)
app.add_middleware(HTTPLoggerMiddleware)




@app.middleware("http")
async def log_request_time(request: Request, call_next):
    start = perf_counter()

    response = await call_next(request)

    duration_ms = (perf_counter() - start) * 1000

    logger.info(
        "%s %s | status=%d | duration=%.2f ms",
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )

    return response


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
                # The key appears next to mask tokens in the choice scaffold;
                # it must tokenize without producing a mask.
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


def letter_token_ids(allowed) -> dict[str, int]:
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


def build_local(request: DecisionRequest) -> tuple[TaggedPrompt, list[Slot]]:
    """Keep the local question and its answer slots in one attention group."""
    parts = [
        "Answer using the supplied context. Choose only from the allowed labels.",
        "Do not explain your answer.",
    ]
    if request.context:
        parts.append(f"\nContext:\n{request.context}")
    question_start = len(parts)
    parts.append(f"\nQuestion:\n{request.question}")

    if request.type == "single_choice":
        parts.append("\nChoose exactly ONE option:")
        parts.extend(f"{LETTERS[i]}: {option.text}" for i, option in enumerate(request.options))
        parts.append("\nReturn only the letter of the selected option.")
        slots = [Slot("\nAnswer:", list(LETTERS[:len(request.options)]))]
    elif request.type == "boolean":
        parts.extend(["\nAnswer the statement as true or false.", "A: True", "B: False"])
        slots = [Slot("\nAnswer:", ["A", "B"])]
    else:
        parts.append("\nFor EACH option below, decide independently whether it should be selected.")
        parts.extend(f"Option {i}: {option.text}" for i, option in enumerate(request.options, 1))
        parts.extend([
            "\nFor each option, answer:",
            "A: Yes, select this option.",
            "B: No, do not select this option.",
            "Answer every option. Multiple options may be selected.",
        ])
        slots = [Slot(f"\nThe option {i} must be selected?", ["A", "B"])
                 for i in range(1, len(request.options) + 1)]
    # All local answer slots follow the same question/rubric. Multiple-choice
    # slots share this boundary and are inserted consecutively in option order.
    prompt = tag_jev_prompt(parts, [(question_start, 1)])
    for slot in slots:
        slot.insertion_offset = prompt.answer_spans[0][1]
    return prompt, slots


def build_jev(
    request: JevRequest, *, first_question_number: int = 1,
    force_binary_choice: bool = False,
) -> tuple[TaggedPrompt, list[Slot], list[QuestionPlan]]:
    """Build a shared prompt and constrained answer slots for all questions."""
    parts = [
        "Evaluate state. Answer each numbered question from its own rubric.",
        "Output fields are fixed. Answer only with the permitted options; no explanations.",
        f"\nstate:\n{as_text(request.state)}",
    ]
    slots: list[Slot] = []
    plans: list[QuestionPlan] = []
    question_part_starts: list[tuple[int, int]] = []
    slot_span_indices: list[int] = []
    next_group_id = first_question_number

    for number, (qid, q) in enumerate(
        request.questions.items(), start=first_question_number,
    ):
        # Include the heading itself, not just the rubric, in this question's
        # attention group. Record structural offsets instead of searching text.
        first_slot = len(slots)
        if q.type == "choice" and (
            force_binary_choice or len(q.criteria) >= CHOICE_BINARY_MIN_OPTIONS
        ):
            keys = list(q.criteria)
            for option_number, key in enumerate(keys, start=1):
                question_part_starts.append((len(parts), next_group_id))
                slot_span_indices.append(len(question_part_starts) - 1)
                next_group_id += 1
                description = q.criteria[key]
                detail = key if description is None else f"{key}: {as_text(description)}"
                parts.extend([
                    f"\nQuestion {number} (choice), option {option_number}: {as_text(q.instructions)}",
                    f"Candidate: {detail}",
                    "According to the state, is this candidate the best answer?",
                    "A: Yes / True", "B: No / False",
                ])
                slots.append(Slot(f"\nQuestion {number}, option {option_number}: Answer:", ["A", "B"]))
            plans.append(QuestionPlan(
                qid, "choice", list(range(first_slot, len(slots))), keys, [], "binary_ranking",
            ))
            continue

        question_part_starts.append((len(parts), next_group_id))
        slot_span_indices.append(len(question_part_starts) - 1)
        next_group_id += 1
        parts.append(f"\nQuestion {number} ({q.type}): {as_text(q.instructions)}")
        if q.type == "noul":
            parts.extend(["A: Yes / True", "B: No / False"])
            if q.criteria:
                for truth in ("true", "false"):
                    if truth in q.criteria:
                        parts.append(f"{truth}: {as_text(q.criteria[truth])}")
            slots.append(Slot(f"\nQuestion {number}: Answer:", ["A", "B"]))
            plans.append(QuestionPlan(qid, "noul", [first_slot], [], [], "binary"))
        elif q.type == "score":
            levels = list(q.criteria)
            parts.append("Choose the level which best matches the state:")
            for i, level in enumerate(levels):
                parts.append(f"{LETTERS[i]}: {as_text(level)}")
            slots.append(Slot(f"\nQuestion {number}: Best matching level? Answer:", list(LETTERS[:len(levels)])))
            plans.append(QuestionPlan(qid, "score", [first_slot], [], levels, "levels"))


        else:
            criteria = q.criteria
            keys = list(criteria)
            # One mask per question regardless of option count: labels A..Z
            # then single-token pairs (AA, AB, ...) cover up to 255 options.
            labels = choice_labels(len(keys))
            parts.append("Choose exactly ONE alternative")
            for label, key in zip(labels, keys, strict=True):
                description = criteria[key]
                detail = key if description is None else f"{key}: {as_text(description)}"
                parts.append(f"{label}: {detail}")
            slots.append(Slot(f"\nQuestion {number}: Answer:", labels))
            plans.append(QuestionPlan(qid, "choice", [first_slot], keys, [], "labels"))

    prompt = tag_jev_prompt(parts, question_part_starts)
    # Only the assistant copy receives answer masks. Its question/rubric and
    # the corresponding user copy share one group, including the answer slot.
    for slot, span_index in zip(slots, slot_span_indices, strict=True):
        _, end, group_id = prompt.answer_spans[span_index]
        slot.insertion_offset = end
        slot.group_id = group_id
    return prompt, slots, plans


def tag_jev_prompt(
    parts: list[str], question_part_starts: list[tuple[int, int]],
) -> TaggedPrompt:
    """Prefill an open assistant turn with question/rubric blocks.

    The user supplies instructions, state and questions. The assistant repeats
    just the questions/rubrics, with answer slots inserted later at their ends.
    Both copies of a question share an attention group, preventing the user
    copy from becoming an unrestricted route to another question's content.
    """
    content = "\n".join(parts)
    # The real chat template opens the assistant role (and handles any thinking
    # prefix). Append prefilled content there without closing the assistant turn
    # or placing another generation prefix after the answer masks.
    rendered = format_prompt(parts)
    if not content or rendered.count(content) != 1:
        raise ValueError("Chat template must preserve the user content exactly once")
    content_start = rendered.index(content)
    part_offsets = []
    offset = 0
    for part in parts:
        part_offsets.append(offset)
        offset += len(part) + 1

    first_question_offset = part_offsets[question_part_starts[0][0]]
    assistant_content = content[first_question_offset:]
    assistant_start = len(rendered) + 1  # The separating newline below.
    spans = []
    answer_spans = []
    for i, (part_index, group_id) in enumerate(question_part_starts):
        start = part_offsets[part_index]
        end = (
            part_offsets[question_part_starts[i + 1][0]]
            if i + 1 < len(question_part_starts)
            else len(content)
        )
        # Use construction offsets, never a search for user-controlled headings.
        spans.append((content_start + start, content_start + end, group_id))
        answer_spans.append((
            assistant_start + start - first_question_offset,
            assistant_start + end - first_question_offset,
            group_id,
        ))
    return TaggedPrompt(
        rendered + "\n" + assistant_content,
        spans + answer_spans,
        answer_spans,
    )


def validate_group_size() -> None:
    if (
        type(MAX_QUESTIONS_PER_PROMPT) is not int
        or MAX_QUESTIONS_PER_PROMPT < 1
    ):
        raise RuntimeError("MAX_QUESTIONS_PER_PROMPT must be a positive integer")


def build_jev_groups(
    request: JevRequest,
) -> list[tuple[TaggedPrompt, list[Slot], list[QuestionPlan]]]:
    """Preserve state, rubrics, IDs and order; only split the questions.

    Group size is never taken from the request. Slot indices remain local
    to each group; responses must be assembled before combining groups.
    The token limit is still enforced per prompt, without truncation.
    """
    validate_group_size()
    items = list(request.questions.items())
    groups = []
    pending: list[tuple[str, JevQuestion]] = []
    pending_start = 1

    def flush_pending() -> None:
        nonlocal pending
        if pending:
            group_request = request.model_copy(update={"questions": dict(pending)})
            groups.append(build_jev(group_request, first_question_number=pending_start))
            pending = []

    for number, (qid, question) in enumerate(items, start=1):
        if question.type == "choice" and len(question.criteria) > 30:
            flush_pending()
            options = list(question.criteria.items())
            chunk_count = (len(options) + BINARY_CHOICE_CHUNK_SIZE - 1) // BINARY_CHOICE_CHUNK_SIZE
            base_size, extra = divmod(len(options), chunk_count)
            start = 0
            for chunk_index in range(chunk_count):
                end = start + base_size + (chunk_index < extra)
                chunk = dict(options[start:end])
                start = end
                for ordered in (chunk, dict(reversed(list(chunk.items())))):
                    chunk_question = question.model_copy(update={"criteria": ordered})
                    chunk_request = request.model_copy(update={"questions": {qid: chunk_question}})
                    groups.append(build_jev(
                        chunk_request, first_question_number=number, force_binary_choice=True,
                    ))
            continue
        if question.type == "choice" and len(question.criteria) >= CHOICE_BINARY_MIN_OPTIONS:
            flush_pending()
            for ordered in (question.criteria, dict(reversed(list(question.criteria.items())))):
                ordered_question = question.model_copy(update={"criteria": ordered})
                single_request = request.model_copy(update={"questions": {qid: ordered_question}})
                groups.append(build_jev(single_request, first_question_number=number))
            continue
        if not pending:
            pending_start = number
        pending.append((qid, question))
        if len(pending) == MAX_QUESTIONS_PER_PROMPT:
            flush_pending()
    flush_pending()
    return groups


def token_question_group(
    text: str, start: int, end: int,
    question_spans: list[tuple[int, int, int]],
) -> int:
    """Assign one token to a question, tolerating merged boundary whitespace.

    BPE can encode both separator newlines as one token, whose character
    interval then intersects two adjacent question spans. Whitespace overlap
    is not evidence that the token contains content from both questions.
    """
    overlapping = [
        (left, right, group) for left, right, group in question_spans
        if start < end and start < right and end > left
    ]
    if not overlapping:
        return 0
    if len(overlapping) == 1:
        return overlapping[0][2]

    content_groups = {
        group for left, right, group in overlapping
        if text[max(start, left):min(end, right)].strip()
    }
    if len(content_groups) > 1:
        # Never silently expose a token carrying another question's content.
        raise ValueError("A token contains non-whitespace content from two question groups")
    if content_groups:
        return next(iter(content_groups))
    # Pure separator: keep it private to the following question. Do not
    # promote a question-boundary token into the shared-state group.
    return overlapping[-1][2]


def build_masked_input(
    prompt: str | TaggedPrompt, slots: list[Slot],
) -> tuple[torch.Tensor, list[int], torch.Tensor | None]:
    """Interleave question text and answer slots, then align token groups.

    Result: user instructions/state/questions, assistant question 1 + rubric +
    label + MASK, question 2 + rubric + label + MASK, ... (open assistant turn).
    positions contains token indices;
    insertion_offset contains character offsets in the original prompt text.
    """
    text = prompt.text if isinstance(prompt, TaggedPrompt) else prompt
    use_groups = USE_SLOT_ATTENTION and isinstance(prompt, TaggedPrompt)
    groups = [] if use_groups else None
    ids = []

    def append_text(segment: str, offset: int) -> None:
        if not segment:
            return
        if use_groups:
            encoded = tokenizer(
                segment, add_special_tokens=False, return_offsets_mapping=True,
            )
            segment_ids = list(encoded["input_ids"])
            # Tokenizer offsets are local to this segment. Translate them back
            # to the original prompt before finding each token's question group.
            groups.extend(
                token_question_group(text, start + offset, end + offset, prompt.question_spans)
                for start, end in encoded["offset_mapping"]
            )
        else:
            segment_ids = tokenizer.encode(segment, add_special_tokens=False)
        # Only placeholders created below may become answer masks. A literal
        # mask in user text would introduce an unregistered prediction target.
        if MASK_ID in segment_ids:
            raise HTTPException(status_code=422, detail="Input text must not contain the model mask token")
        ids.extend(segment_ids)

    positions = []
    cursor = 0
    for slot in slots:
        # Built-in builders supply an inline boundary. None retains the legacy
        # append-at-end behavior for manually constructed slots/plain prompts.
        boundary = len(text) if slot.insertion_offset is None else slot.insertion_offset
        if not cursor <= boundary <= len(text):
            raise ValueError("Slot insertion offsets must follow prompt order")
        # Encode only up to this answer location; later questions are appended
        # after its mask. Splitting here prevents tokens crossing an insertion.
        append_text(text[cursor:boundary], cursor)
        cursor = boundary
        label_ids = tokenizer.encode(slot.label, add_special_tokens=False)
        if MASK_ID in label_ids:
            raise HTTPException(status_code=422, detail="Input text must not contain the model mask token")
        ids.extend(label_ids)
        # Insert the model's mask ID directly, independently of how its textual
        # spelling would tokenize. Record its exact index for decoder projection.
        positions.append(len(ids))
        ids.append(MASK_ID)
        if groups is not None:
            # Both the answer label and MASK belong to their question: attention
            # permissions cover the slot in both query and key directions.
            groups.extend([slot.group_id] * (len(label_ids) + 1))
    # Preserve trailing template tokens after the last inline answer.
    if cursor < len(text):
        append_text(text[cursor:], cursor)
    if len(ids) > MAX_INPUT_TOKENS:
        raise HTTPException(
            status_code=422,
            detail=f"Input with {len(ids)} tokens exceeds MAX_INPUT_TOKENS={MAX_INPUT_TOKENS}",
        )
    device = model.get_input_embeddings().weight.device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    group_ids = None
    if groups is not None:
        if len(groups) != len(ids):
            raise ValueError("Token/group alignment failed")
        group_ids = torch.tensor(groups, dtype=torch.long, device=device)
        present_groups = set(group_ids.tolist())
        if any(slot.group_id not in present_groups or slot.group_id < 0 for slot in slots):
            raise ValueError("Invalid slot group")
    return tokens, positions, group_ids


def evaluate(
    prompt: str | TaggedPrompt,
    slots: list[Slot],
) -> tuple[list[str], list[list[float]], list[list[float]], int, int]:
    """Decode one prompt and retain per-slot logits and probabilities."""
    gpu_stats("Antes da inferência")

    torch.cuda.reset_peak_memory_stats()



    input_ids, positions, group_ids = build_masked_input(prompt, slots)
    # Decode the assembled token stream: this includes the open assistant turn,
    # inline answer labels and the actual mask tokens seen by the model.
    masked_ids = input_ids[0].tolist()
    logger.info("PROMPT | %s", prompt.text if isinstance(prompt, TaggedPrompt) else prompt)
    logger.info("ASSISTANT INPUT | %s", tokenizer.decode(masked_ids, skip_special_tokens=False))
    # Map application labels to candidate token IDs. Attention determines which
    # input tokens are visible; candidates independently restrict output values.
    token_ids = letter_token_ids(dict.fromkeys(label for slot in slots for label in slot.allowed))
    result = decode(
        model, input_ids, positions,
        [[token_ids[letter] for letter in slot.allowed] for slot in slots],
        mask_id=MASK_ID, mode=DECODER_MODE, threshold=DECODER_THRESHOLD,
        group_ids=group_ids,
    )

    logger.info(
        "DECODER | mode=%s | "
        "masks=%d | forwards=%d",
        DECODER_MODE,
        len(positions),
        result.forward_passes,
    )


    labels = [slot.allowed[winner] for slot, winner in zip(slots, result.winners, strict=True)]
    filled_ids = masked_ids.copy()
    for position, label in zip(positions, labels, strict=True):
        filled_ids[position] = token_ids[label]
    logger.info(
        "SLOTS | %s",
        json.dumps([
            {"position": position, "label": slot.label, "group_id": slot.group_id,
             "allowed": slot.allowed, "selected": label,
             "logits": dict(zip(slot.allowed, slot_logits, strict=True)),
             "probabilities": dict(zip(slot.allowed, slot_probs, strict=True))}
            for position, slot, label, slot_logits, slot_probs in zip(
                positions, slots, labels, result.logits, result.probabilities, strict=True,
            )
        ], ensure_ascii=False),
    )
    logger.info("ASSISTANT RESULT | %s", tokenizer.decode(filled_ids, skip_special_tokens=False))

    gpu_stats("Depois da inferência")

    return labels, result.logits, result.probabilities, input_ids.shape[1], result.forward_passes


async def evaluate_jev_groups(
    groups: list[tuple[TaggedPrompt, list[Slot], list[QuestionPlan]]],
) -> tuple[dict[str, dict], int, int]:
    """Evaluate sequentially, assemble locally, then merge by question ID.

    No cross-group slot offsets are needed: each plan is consumed alongside
    its own decoder outputs. Multi-slot binary-ranking questions stay intact.
    Usage counts include the repeated state in every evaluated group.
    """
    answers: dict[str, dict] = {}
    binary_choices: dict[str, tuple[list[str], dict[str, list[float]]]] = {}
    question_order: list[str] = []
    total_tokens = 0
    total_passes = 0
    async with inference_lock:
        for group_number, (prompt, slots, plans) in enumerate(groups, start=1):
            logger.info(
                "JEV GROUP | group=%d/%d | questions=%d | masks=%d | decoder=%s | slot_attention=%s",
                group_number, len(groups), len(plans), len(slots), DECODER_MODE, USE_SLOT_ATTENTION,
            )
            labels, raw, probs, n_tokens, passes = evaluate(prompt, slots)
            validate_decoded_results(plans, labels, raw, probs)
            ordinary_plans = []
            for plan in plans:
                if plan.question_id not in question_order:
                    question_order.append(plan.question_id)
                if plan.kind == "choice" and plan.mode == "binary_ranking":
                    keys, samples = binary_choices.setdefault(plan.question_id, ([], {}))
                    for key, index in zip(plan.option_keys, plan.slot_indices, strict=True):
                        if key not in samples:
                            keys.append(key)
                            samples[key] = []
                        samples[key].append(raw[index][0] - raw[index][1])
                else:
                    ordinary_plans.append(plan)
            group_answers = create_jev_answers(ordinary_plans, labels, raw, probs)
            if answers.keys() & group_answers.keys():
                raise ValueError("Duplicate question IDs across Jev groups")
            answers.update(group_answers)
            total_tokens += n_tokens
            total_passes += passes
    for qid, (keys, samples) in binary_choices.items():
        if qid in answers or any(len(samples[key]) != 2 for key in keys):
            raise ValueError(f"Incomplete bidirectional choice result for question {qid!r}")
        margins = [sum(samples[key]) / 2 for key in keys]
        answers[qid] = rank_binary_choice(keys, margins)
        top = sorted(zip(keys, margins, strict=True), key=lambda item: item[1], reverse=True)[:5]
        yes_votes = sum(margin > 0 for values in samples.values() for margin in values)
        logger.info(
            "CHOICE RANK | question=%s | winner=%s | yes_votes=%d/%d | top_yes_no_margins=%s",
            qid, answers[qid]["choice"], yes_votes, sum(map(len, samples.values())), top,
        )
    return {qid: answers[qid] for qid in question_order}, total_tokens, total_passes


def validate_decoded_results(
    plans: list[QuestionPlan],
    labels: list[str],
    raw: list[list[float]],
    probs: list[list[float]],
) -> None:
    """Validate plan/result alignment before constructing Jev responses."""
    result_sizes = {
        "labels": len(labels),
        "logits": len(raw),
        "probabilities": len(probs),
    }
    if len(set(result_sizes.values())) != 1:
        raise ValueError(f"Decoder result arrays have different sizes: {result_sizes}")

    result_count = len(labels)
    for plan in plans:
        if not plan.slot_indices:
            raise ValueError(f"Question {plan.question_id!r} has no answer slots")

        if plan.kind == "noul":
            expected_slots = 1
            expected_candidates = 2
        elif plan.kind == "score":
            expected_slots = 1
            expected_candidates = len(plan.levels)
        elif plan.kind == "choice" and plan.mode == "labels":
            expected_slots = 1
            expected_candidates = len(plan.option_keys)
        elif plan.kind == "choice" and plan.mode == "binary_ranking":
            expected_slots = len(plan.option_keys)
            expected_candidates = 2
        else:
            raise ValueError(
                f"Unknown plan type for question {plan.question_id!r}: "
                f"kind={plan.kind!r}, mode={plan.mode!r}"
            )

        if len(plan.slot_indices) != expected_slots:
            raise ValueError(
                f"Slot count mismatch for question {plan.question_id!r}: "
                f"expected {expected_slots}, received {len(plan.slot_indices)}"
            )

        for slot_index in plan.slot_indices:
            if not 0 <= slot_index < result_count:
                raise ValueError(
                    f"Invalid slot index {slot_index} for question "
                    f"{plan.question_id!r}; decoder returned {result_count} slots"
                )

            probability_count = len(probs[slot_index])
            if probability_count not in {0, expected_candidates}:
                raise ValueError(
                    f"Probability mismatch for question {plan.question_id!r}: "
                    f"expected {expected_candidates}, received "
                    f"{probability_count}, slot={slot_index}"
                )

            logit_count = len(raw[slot_index])
            if logit_count not in {0, expected_candidates}:
                raise ValueError(
                    f"Logit mismatch for question {plan.question_id!r}: "
                    f"expected {expected_candidates}, received {logit_count}, "
                    f"slot={slot_index}"
                )


def rank_binary_choice(option_keys: list[str], margins: list[float]) -> dict:
    """Turn independent yes/no logit margins into one choice response."""
    if not option_keys or len(option_keys) != len(margins):
        raise ValueError("Binary choice options and margins must align")
    winner = max(range(len(margins)), key=margins.__getitem__)
    distribution = torch.tensor(margins, dtype=torch.float32).softmax(dim=0).tolist()
    return {
        "type": "choice",
        "choice": option_keys[winner],
        "confidence": max(distribution),
        "probabilities": dict(zip(option_keys, distribution, strict=True)),
    }


async def evaluate_pairwise_choice(
    request: JevRequest, question_id: str, question: JevQuestion, number: int,
) -> tuple[dict, int, int]:
    """Choose among many options using two-order pairwise comparisons.

    Both orders are scored before advancing a candidate. This removes the
    model's strong preference for the second position in a two-option prompt.
    Defeat margins form a tree rooted at the winner; path sums give one score
    per option for the required response distribution.
    """
    keys = list(question.criteria)
    contenders = keys.copy()
    defeated: dict[str, list[tuple[str, float]]] = {key: [] for key in keys}
    total_tokens = 0
    total_passes = 0

    async with inference_lock:
        while len(contenders) > 1:
            next_round = []
            for index in range(0, len(contenders) - 1, 2):
                left, right = contenders[index:index + 2]
                margins = []
                for ordered in ((left, right), (right, left)):
                    pair_criteria = {key: question.criteria[key] for key in ordered}
                    pair_question = question.model_copy(update={"criteria": pair_criteria})
                    pair_request = request.model_copy(update={"questions": {question_id: pair_question}})
                    prompt, slots, _ = build_jev(pair_request, first_question_number=number)
                    _, raw, _, n_tokens, passes = evaluate(prompt, slots)
                    if len(raw) != 1 or len(raw[0]) != 2:
                        raise ValueError("Pairwise choice must return two logits")
                    margins.append(raw[0][0] - raw[0][1] if ordered[0] == left
                                   else raw[0][1] - raw[0][0])
                    total_tokens += n_tokens
                    total_passes += passes
                margin = sum(margins) / 2
                winner, loser = (left, right) if margin >= 0 else (right, left)
                defeated[winner].append((loser, abs(margin)))
                next_round.append(winner)
            if len(contenders) % 2:
                next_round.append(contenders[-1])
            contenders = next_round

    winner = contenders[0]
    scores = {winner: 0.0}
    pending = [winner]
    while pending:
        parent = pending.pop()
        for child, margin in defeated[parent]:
            scores[child] = scores[parent] - margin
            pending.append(child)
    if len(scores) != len(keys):
        raise ValueError("Incomplete pairwise choice tournament")
    distribution = torch.tensor([scores[key] for key in keys], dtype=torch.float32).softmax(0).tolist()
    answer = {
        "type": "choice", "choice": winner,
        "confidence": max(distribution),
        "probabilities": dict(zip(keys, distribution, strict=True)),
    }
    logger.info(
        "CHOICE TOURNAMENT | question=%s | winner=%s | matches=%d | top_scores=%s",
        question_id, winner, len(keys) - 1,
        sorted(scores.items(), key=lambda item: item[1], reverse=True)[:5],
    )
    return answer, total_tokens, total_passes


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
    validate_decoded_results(plans, labels, raw, probs)

    answers: dict[str, dict] = {}
    for plan in plans:
        if plan.kind == "noul":
            slot = plan.slot_indices[0]
            selected = labels[slot]
            if selected not in {"A", "B"}:
                raise ValueError(
                    f"Invalid noul label {selected!r} for question "
                    f"{plan.question_id!r}"
                )
            yes_probability = probs[slot][0] if probs[slot] else (1.0 if selected == "A" else 0.0)
            answers[plan.question_id] = {
                "type": "noul",
                "noul": yes_probability,
            }
        elif plan.kind == "score":
            slot = plan.slot_indices[0]
            selected = labels[slot]
            winner = LETTERS.index(selected)
            if winner >= len(plan.levels):
                raise ValueError(
                    f"Invalid score label {selected!r} for question "
                    f"{plan.question_id!r} with {len(plan.levels)} levels"
                )
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
                winner = choice_labels(len(plan.option_keys)).index(selected)
                if winner >= len(plan.option_keys):
                    raise ValueError(
                        f"Invalid choice label {selected!r} for question "
                        f"{plan.question_id!r} with "
                        f"{len(plan.option_keys)} options"
                    )
                distribution = probs[slot] if probs[slot] else [1.0 if i == winner else 0.0 for i in range(len(plan.option_keys))]
            elif plan.mode == "binary_ranking":
                if any(len(raw[index]) != 2 for index in plan.slot_indices):
                    raise ValueError(f"Missing yes/no logits for question {plan.question_id!r}")
                margins = [raw[index][0] - raw[index][1] for index in plan.slot_indices]
                answers[plan.question_id] = rank_binary_choice(plan.option_keys, margins)
                continue
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
        pairwise = {
            qid: (number, question)
            for number, (qid, question) in enumerate(request.questions.items(), start=1)
            if question.type == "choice" and len(question.criteria) >= CHOICE_BINARY_MIN_OPTIONS
        }
        ordinary = {qid: question for qid, question in request.questions.items() if qid not in pairwise}
        groups = build_jev_groups(request.model_copy(update={"questions": ordinary})) if ordinary else []
        logger.info(
            "JEV | questions=%d | groups=%d | max_questions_per_prompt=%d | "
            "request_isolated=%s (ignored)",
            len(request.questions), len(groups), MAX_QUESTIONS_PER_PROMPT,
            request.isolated,
        )
        answers, n_tokens, passes = await evaluate_jev_groups(groups)
        for qid, (number, question) in pairwise.items():
            answer, choice_tokens, choice_passes = await evaluate_pairwise_choice(
                request, qid, question, number,
            )
            answers[qid] = answer
            n_tokens += choice_tokens
            passes += choice_passes
        answers = {qid: answers[qid] for qid in request.questions}
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
