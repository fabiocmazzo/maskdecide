# MaskDecide architecture

This document describes how MaskDecide turns HTTP requests into typed decisions
using a diffusion language model, and where the current design breaks down.

## Overview

MaskDecide is a FastAPI service that loads a single Fast-dLLM v2 checkpoint and
exposes two POST endpoints:

- `POST /decide` — a local, ergonomic API for booleans, single-choice, and
  multiple-choice decisions.
- `POST /v1/systemone` — an experimental adapter inspired by TypeSafe AI's Jev
  System One interface. It accepts a shared `state` and a map of `questions`
  with types `noul`, `choice`, and `score`.

Both endpoints follow the same pipeline:

1. Validate the request and normalize structured fields to text.
2. Build a prompt containing the context/state, questions, rubrics, and answer
   slots.
3. Tokenize the prompt and append one mask token per answer slot.
4. Run the constrained decoder to fill each mask with an allowed label.
5. Map labels back to application values and construct the JSON response.

The structural guarantee is that every answer field exists and has the correct
type. The guarantee is syntactic, not semantic: the model can still pick the
wrong label.

## Model loading and quantization

At startup the service loads `Efficient-Large-Model/Fast_dLLM_v2_7B` with
bitsandbytes 4-bit NF4 quantization and bfloat16 compute. The entire model is
placed on GPU 0 (`device_map={"": 0}`). Loading uses `trust_remote_code=True`
because Fast-dLLM v2 ships its own modeling code.

Key constants:

- `MASK_ID = 151665` — the token ID of `|<MASK>|` in the Fast-dLLM v2 tokenizer.
- `BLOCK_SIZE = 32` — the model's block-causal attention window.
- `MAX_INPUT_TOKENS = 4096` — hard limit for the encoded prompt.

A single `asyncio.Lock` serializes all inference. The lock is async so that
waiting requests do not block the event loop or starve the threadpool that
serves `/health` and `/docs`.

## Prompt construction

### Shared prompt (default)

`build_jev` concatenates:

1. A fixed instruction header.
2. `Shared state: <serialized state>`.
3. For each question, in request order:
   - `Question N (<type>): <instructions>`
   - The rubric lines (`A: ...`, `B: ...`, etc.)
   - One or more answer slots.

An answer slot is a short label followed by a mask token, for example:

```
Question 1: Is the answer yes/true? Answer: |<MASK>|
```

The label is chosen so that the mask is always immediately preceded by a fixed
scaffold. This makes the mask position unambiguous and keeps the prefix cache
free of unresolved masks.

### Isolated prompt (`"isolated": true`)

`build_jev_isolated` creates one prompt per question. Each prompt contains the
same state and exactly one question with its rubric and slots. Questions cannot
attend to each other's answer slots, which removes order-dependent interactions
(such as the Hanna/Jeff addressee case) at the cost of roughly N times more
tokens and latency.

### Local `/decide` prompts

`build_local` uses a simpler format: context, question, and then either a
single letter-choice slot (boolean/single_choice) or one yes/no slot per option
(multiple_choice).

## Answer-slot encoding

The critical invariant is that every answer must be representable as a single
token. The allowed labels are always space-prefixed capital letters (` A`,
` B`, ...), which encode as single tokens in the Fast-dLLM v2 tokenizer.

`letter_token_ids` verifies this at startup and raises if a label ever encodes
to more than one token. `build_masked_input` additionally rejects any input
that already contains the mask token ID, so user text cannot inject fake answer
slots.

For choices with 2–26 options, a single slot is enough: the alphabet has 26
letters. For 27–255 options, MaskDecide falls back to `binary_ranking`: each
option gets its own yes/no slot, and the winner is the option with the largest
`logit(A) - logit(B)` margin. This is a heuristic, not a joint categorical
distribution over the options.

## Constrained decoding

The decoder (`src/maskdecide/decoding.py`) implements two modes.

### `one_pass`

Runs the model once over the full masked input and reads the logits at
`position - 1` for each mask. Fast-dLLM v2 is shifted: the logits at position
`p-1` predict the token at position `p`. This mode is fast but evaluates all
slots against the same unresolved context.

### `iterative` (default)

The iterative decoder resolves 32-token blocks from left to right:

1. **Prefill**: if the current block starts beyond the cached prefix, run the
   model on the clean prefix and update the KV cache. Only complete, mask-free
   blocks are ever cached.
2. **Boundary slot**: if the first token of the block is a mask, its prediction
   comes from the previous block's final logits.
3. **Block loop**: for the remaining masks in the block, run the model on the
   active block with `update_past_key_values=False` and `logits_to_keep` set to
   the prediction positions. Compute the restricted softmax over each slot's
   allowed labels.
4. **Commit**: accept all predictions whose restricted probability is at least
   `DECODER_THRESHOLD` (default 0.9). If none qualifies, commit the strongest
   remaining prediction. This guarantees progress.
5. Repeat until the block is resolved, then advance to the next block.

The maximum number of model calls is bounded by twice the number of answer
slots, including prefills. Each slot's reported logits and probabilities are
captured **before** its answer is committed; rescoring a filled slot would leak
the answer through bidirectional attention within the block.

## Response construction

`create_jev_answers` translates the winning labels and their pre-commit
restricted softmax probabilities into the Jev response shapes:

- `noul`: the probability of the "yes" label (`A`).
- `choice`: the selected key, the full restricted-softmax distribution over
  options, and `confidence = max(probability)`.
- `score`: the probability-weighted level index, the distribution, `legend`,
  and `confidence = max(probability)`.

For `binary_ranking`, the independent yes-probabilities are normalized to sum
to 1. This is a response convention, not a calibrated joint distribution.

`usage` reports `input_tokens`, `output_tokens: 0`, and `forward_passes`, the
actual number of model calls made by the decoder.

## Concurrency model

- The lifespan loads the model before the server accepts requests.
- `asyncio.Lock` serializes inference; endpoints are `async def` and hold the
  lock with `async with`.
- Because the lock is async, a queued request does not block the event loop.
  `/health` remains responsive while inference is running.
- There is no cross-request batching. Each HTTP request triggers its own
  decoder run.

## Validation and safety

- Request sizes are capped (`MAX_INPUT_TOKENS`, `MAX_QUESTIONS`, option-count
  limits).
- Option IDs and criteria keys must match `^[a-zA-Z][a-zA-Z0-9_]*$` (max 40
  characters). This keeps their tokenization next to answer masks predictable.
- Input text that tokenizes to the literal mask ID is rejected with 422.
- Non-finite candidate logits raise an internal error instead of silently
  selecting an answer.
- Optional bearer-token authentication is available via `LOCAL_API_KEY`.

## Limitations

### Model capability

The decoder can only extract what the model knows. Fast-dLLM v2 still struggles
with negation, temporal reasoning, numeric boundaries, and some addressee
resolution. Switching from 1.5B to 7B improved several smoke cases, but the
remaining failures are model limitations, not decoder bugs.

### Shared-prompt interaction

By default all questions share one prompt. Later answer slots can attend to
earlier unresolved masks inside the same block, and the presence or order of
other questions can change answers. The `isolated` flag removes this
interaction but costs more tokens and latency.

### Large-choice ranking

`binary_ranking` compares independent yes/no margins from different positions
and conditioning contexts. These margins do not form a joint categorical
distribution. The normalized probabilities returned for large choices are a
heuristic response format, not evidence that the model jointly compared all
options.

### Calibration

Restricted softmax probabilities measure the model's preference among the
allowed labels, not factual certainty. They are not calibrated and should not
be used as confidence thresholds for downstream automation.

### Latency and throughput

- Iterative decoding can cost up to 2× the number of slots in model calls.
  A 27-option choice required 42 forward passes in one measurement.
- The inference lock serializes all requests; throughput is one request at a
  time.
- Isolated mode multiplies token count and latency by the number of questions.

### Structural constraints

- Answer labels must remain single tokens. Any tokenizer or model change that
  breaks this invariant will fail loudly at startup.
- The 4096-token input limit can reject requests well below the 128-question
  cap when options are numerous or verbose.
- `output_tokens` is always 0 because the adapter does not generate prose; the
  real compute cost is reported in `forward_passes`.

## Failure modes that are explicitly handled

- **Unregistered masks**: rejected before any model call.
- **Non-finite logits**: raise `ValueError` and surface as HTTP 500.
- **Mask token in user input**: rejected with 422.
- **Invalid criteria keys**: rejected with 422.
- **Block-boundary masks**: handled by reading the previous block's final
  logits; covered by unit tests.
- **Cache staleness**: only complete, mask-free blocks are cached; covered by
  unit tests.

## What is not handled

- No calibrated confidence or abstention mechanism.
- No automatic retry or fallback when the model is uncertain.
- No request batching or multi-GPU support.
- No CPU inference path.
- No guarantee that the selected answer is factually correct or satisfies
  application business rules.
