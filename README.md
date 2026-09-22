# MaskDecide

Local typed decisions powered by a diffusion language model.

MaskDecide turns context and questions into booleans, choices, and discrete scores
using [Fast-dLLM v2 7B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B).
It fills masked answer positions with a constrained iterative decoder, restricts
each answer to allowed labels, and assembles the result into a typed HTTP
response. The original single-pass decoder is available for comparison.

This is an experimental open-source project demonstrating how existing diffusion
models can support structured decision tasks locally. It is not intended to
compete with Jev or claim equivalent intelligence, speed, or calibration. The
Jev-inspired HTTP adapter makes the approach easier to explore with a familiar
interface. MaskDecide is not affiliated with or endorsed by TypeSafe AI.

**Status:** working prototype. There are no robust comparative benchmarks yet.
Typed output guarantees constrain the response format, not the correctness of a
decision. Evaluate accuracy on your own tasks before relying on the results.
See [docs/architecture.md](docs/architecture.md) for a detailed description of
the design and its limitations.

## Requirements

- Python 3.13 or later.
- [uv](https://docs.astral.sh/uv/getting-started/installation/) and GNU Make.
- A CUDA-capable NVIDIA GPU and a compatible driver/PyTorch installation.
- Enough GPU memory for the model and your input; memory requirements increase
  with context length and question count. No minimum VRAM has been established.
- Internet access on first startup to download the model and tokenizer from
  Hugging Face. Cached assets can be reused afterward.

The current server requires CUDA and loads the model in 4-bit NF4 quantization
(bitsandbytes) with `bfloat16` compute on GPU 0. CPU-only inference is not
supported. Loading uses `trust_remote_code=True` to execute the model
implementation from its upstream repository.

## Quick start

From the repository directory:

```sh
make install
make serve
```

The server binds to `127.0.0.1:8000`. The first startup can take longer while model
assets download and load. Use a single worker: each worker would load another
model instance.

```sh
curl http://127.0.0.1:8000/health
```

Interactive API documentation is available at
<http://127.0.0.1:8000/docs> after startup.

Equivalent commands without Make:

```sh
uv sync --locked
uv run --locked maskdecide
```

Use `make serve PORT=8080` or `uv run --locked maskdecide --port 8080` for a
different port. The original `uv run uvicorn decision_api_jev:app` entry point
also remains available from the repository root.

## Local decisions API

`POST /decide` accepts `context`, `question`, `type`, and optional `options`.

| Type | Options | Answer |
| --- | --- | --- |
| `boolean` | Must be empty or omitted | Boolean |
| `single_choice` | 2–20 options | One option ID |
| `multiple_choice` | 2–20 options | A list of selected option IDs, possibly empty |

Each option contains a unique `id` and a `text` description. IDs must start with
an ASCII letter and contain only letters, digits, or underscores (40 characters
maximum).

```sh
curl http://127.0.0.1:8000/decide \
  -H 'Content-Type: application/json' \
  -d '{
    "context": "The customer was charged twice for the same order.",
    "question": "Which team should handle this request?",
    "type": "single_choice",
    "options": [
      {"id": "billing", "text": "Payments and invoices"},
      {"id": "support", "text": "Technical support"}
    ]
  }'
```

An illustrative response, assuming the model selects billing:

```json
{
  "type": "single_choice",
  "answer": "billing",
  "model": "Efficient-Large-Model/Fast_dLLM_v2_7B"
}
```

## Experimental Jev HTTP adapter

`POST /v1/systemone` accepts a shared `state` and a map of `questions`. State can
be text, an object, or an array. Instructions can also be text, an object, or an
array. All questions are placed in one prompt.

```sh
curl http://127.0.0.1:8000/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "jev-latest",
    "state": "The customer was charged twice. Login works normally.",
    "questions": {
      "billing_issue": {
        "type": "noul",
        "instructions": "Does the customer report a billing problem?"
      },
      "team": {
        "type": "choice",
        "instructions": "Which team should handle the request?",
        "criteria": {
          "billing": "Payments and invoices",
          "support": "Technical support"
        }
      },
      "urgency": {
        "type": "score",
        "instructions": "Assess urgency using the supplied rubric.",
        "criteria": ["General inquiry", "Customer needs assistance", "Service-wide outage"]
      }
    }
  }'
```

Responses contain `model`, `answers` keyed by question ID, and `usage`.
The returned model ID always identifies Fast-dLLM. The request's `model` field
is accepted for interface compatibility; it does not select a different backend.

| Question | Criteria | Current answer semantics |
| --- | --- | --- |
| `noul` | Optional object with `true`/`false` descriptions | `noul` is the restricted softmax probability of the "yes" label |
| `choice` | Object containing 2–255 option keys and descriptions | One `choice`, restricted-softmax `probabilities`, `confidence` = max probability |
| `score` | Array of 2–10 level descriptions | Probability-weighted `score`, plus `legend`, restricted-softmax `probabilities`, `confidence` = max probability |

**Confidence is not calibrated.** The reported `probabilities` and `confidence`
are the model's restricted softmax over the permitted labels for each answer
slot, computed before that slot's answer is committed. They describe the
model's preference among the allowed labels, not factual certainty, and must
not be used as calibrated confidence thresholds.
The native `/decide` endpoint returns decisions without confidence fields.

Set `"isolated": true` in the request to evaluate each question in its own
prompt. This prevents questions from attending to each other's answer slots
and removes order-dependent interactions, at the cost of increased token count
and latency. The default `false` preserves the shared-prompt behavior.

Compatibility is limited to the implemented request/response shapes. MaskDecide
does not reproduce Jev's probabilistic semantics or guarantee full SDK/API
compatibility. See the [official Jev documentation](https://docs.typesafe.ai/)
for its contract.

## How it works

1. Serialize the context, questions, and rubrics into a prompt.
2. Append labeled answer slots containing mask tokens.
3. Process answer blocks from left to right, caching only completed blocks.
4. Within each block, restrict predictions to allowed single-token labels,
   commit strong predictions, then reevaluate the remaining masks with the
   updated input. If no prediction reaches the scheduling threshold, commit
   the strongest remaining prediction to guarantee progress.
5. Map the labels back to application values and build the JSON response.

Choices with up to 26 options use one answer slot. Choices with 27–255 options use
one yes/no slot per option and select the option with the largest yes/no logit
margin. This is a different selection procedure and needs separate evaluation.
The implementation follows Fast-dLLM v2's token-shift convention: the logits at
the preceding position predict the mask token.

The iterative decoder uses 32-token blocks. A mask at the start of a block uses
the previous block's last logits. It never modifies the context or answer
scaffolding and never caches representations of unresolved masks. Large-choice
ranking uses each slot's logits from immediately before its answer is filled;
rescoring a filled slot could leak the answer through bidirectional attention.
The number of forward calls is bounded by twice the number of answer slots,
including prefix caching. Logs show the decoder mode and actual call count.

The internal restricted softmax schedules which masks to fill and also
produces the response `probabilities` and `confidence`. These values are not
calibrated and do not measure factual accuracy.
With only one answer mask, there are no other answers to refine against, so
iteration alone cannot be expected to fix the decision.

To use the original decoder, start the server with:

```sh
DECODER_MODE=one_pass make serve
```

To explicitly select iterative decoding:

```sh
DECODER_MODE=iterative DECODER_THRESHOLD=0.9 make serve
```

Changing modes requires restarting the server. The iterative implementation
follows the upstream block order and token shift, but is a constrained decoder
for fixed answer slots, not an exact reproduction of upstream text generation.

## Configuration

| Setting | Default | Purpose |
| --- | --- | --- |
| `MAX_INPUT_TOKENS` | `4096` | Limit for the entire encoded prompt, including questions and answer slots |
| `LOCAL_API_KEY` | Unset | Optional bearer token required by both POST endpoints |
| `DECODER_MODE` | `iterative` | `iterative` for blockwise unmasking; `one_pass` for the original decoder |
| `DECODER_THRESHOLD` | `0.9` | Internal scheduling threshold in `(0, 1]`; applies only to iterative decoding |
| CLI `--host` / Make `HOST` | `127.0.0.1` | Bind address |
| CLI `--port` / Make `PORT` | `8000` | Listen port |

Environment variables are read at server startup. An `.env` file is not loaded
automatically. With `LOCAL_API_KEY` set, include
`Authorization: Bearer <your-key>` in POST requests. The smoke test reads the same
environment variable or accepts `--token`. `/health` and `/docs` remain public.
Authentication is disabled by default for local use; configure authentication
and deployment controls before exposing the server beyond localhost.

## Tests

```sh
make test
```

Local tests check request validation, answer mapping, and smoke-test failure
reporting without loading model weights or requiring a GPU. They do not establish
model quality. Decoder tests cover cache validity, block boundaries, constrained
selection, recomputation, and guaranteed progress.

To check actual inference, leave `make serve` running in another terminal:

```sh
make test-smoke
make test-smoke TEST_ARGS='--repeat 3'
make test-smoke TEST_ARGS='--list'
make test-smoke TEST_ARGS='--case "Score boundary"'
```

The suite contains 39 cases and 49 expected decisions per run. It covers
negation, temporal facts, explicit boolean rubrics, missing evidence, routing,
option and question ordering, structured state and instructions, Unicode,
mixed question types, score boundaries, and choices with 26 and 27 options.
`--case` selects case names by a case-insensitive substring; repeat it to match
any of several substrings. `--list` lists matching cases without contacting the
server. Unmatched filters are rejected instead of silently running zero tests.

For another server, use
`uv run --locked python test_jev_api_smoke.py --url http://127.0.0.1:8080`.
The smoke suite exits with a nonzero status for either HTTP/contract errors or
incorrect decisions. Its examples and latency summary are diagnostic only, not
a representative benchmark or evidence of calibration.

With the original `one_pass` decoder and the 1.5B model on an NVIDIA GeForce
RTX 4060 Laptop GPU, all 39 requests returned valid responses, and 37 of 49
decisions matched expectations. Failures included identifying the addressee of
a sentence, negation and temporal facts, a boolean threshold, four score
boundaries, and the 27-option choice. Reordering the addressee questions also
changed the answer about Jeff. The smoke target therefore exited with a failure
status and listed the failed cases and question IDs. These observations describe
this small suite and are not a benchmark or a general accuracy estimate. Failing
cases remain in the suite to make the limitations reproducible.

The iterative decoder subsequently scored 40/49 on the same suite and 8/13 on
additional examples, compared with 37/49 and 8/13 for `one_pass`. Three repeats
gave the same decisions. Warm median decoding time on the smoke suite increased
from approximately 30 ms to 47 ms, excluding HTTP and tokenization. Large choices
can cost substantially more. See [the decoder comparison](docs/decoder-comparison.md)
for methodology, remaining failures, and reproduction commands. These small
diagnostic suites do not establish general accuracy or a production latency SLA.
The current default model is 7B; accuracy and latency characteristics differ
from the 1.5B results above.

## Current limitations

- Accuracy, latency, throughput, and calibration have not been established by a
  robust comparative benchmark.
- Multiple questions share a prompt by default. Set `"isolated": true` to
  evaluate each question in its own prompt; this removes cross-question
  attention at the cost of more tokens and latency.
- Requests are serialized by an async inference lock; there is no request
  batching across HTTP requests.
- A maximum of 128 questions is accepted, but the total token limit can reject
  requests well below that count, especially with many options.
- Logged decoder time includes its model calls and label selection. HTTP latency
  also includes tokenization, queuing, response construction, and transport.
- Iterative decoding can increase latency substantially when there are many
  masked slots, especially for choices with more than 26 options.
- Literal model mask tokens (`|<MASK>|`) in input text are rejected to keep
  answer-slot positions unambiguous and cached prefixes free of masks.
- Criteria keys and option IDs are restricted to ASCII letters, digits, and
  underscores (starting with a letter, 40 characters maximum) so their
  tokenization next to answer masks stays predictable.
- `usage.output_tokens` is zero because the adapter does not emit generated prose;
  answer-slot computation still has a cost. `usage.forward_passes` reports the
  actual number of model calls made by the decoder.
- Restricting answer types does not guarantee factual accuracy or enforce the
  application's business rules.

Useful next steps include evaluating real classification and routing tasks,
testing sensitivity to ordering and ambiguity, investigating calibrated
confidence, and comparing against a small autoregressive model constrained to
single-token answers on the same hardware.

## License and acknowledgments

MaskDecide's original code and documentation are licensed under the
[Apache License 2.0](LICENSE).

Model weights, tokenizer assets, upstream model code, and dependencies remain
subject to their own licenses. They are not relicensed by this project. The
Fast-dLLM v2 7B model card currently declares Apache 2.0; provenance and links
are recorded in [third-party notices](THIRD_PARTY_NOTICES.md). Model assets are
downloaded separately and are not included in this repository.

This project builds on [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM) and its
[v2 research](https://arxiv.org/abs/2509.26328). The adapter's interface is inspired
by [Jev from TypeSafe AI](https://typesafe.ai/).
