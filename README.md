# MaskDecide

MaskDecide is a local FastAPI service that turns state and questions into typed
decisions with the [`nvidia/Nemotron-Labs-Diffusion-3B`](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B)
diffusion model. It fills masked answer slots using a restricted set of token
labels, then maps those labels to booleans, choices, or scores. It exposes a
small local API and an experimental HTTP adapter inspired by Jev System One.
It is an independent prototype, not a Jev model or an implementation endorsed
by TypeSafe AI.

**Status:** experimental. Typed responses guarantee the response shape, not the
correctness of a decision. The probabilities and confidence values are model
preferences or tournament scores, not calibrated estimates of factual accuracy.

## Requirements and setup

- Python 3.13 or later, [uv](https://docs.astral.sh/uv/getting-started/installation/), and GNU Make.
- An NVIDIA GPU with CUDA and a compatible PyTorch installation. CPU-only server
  inference is not supported.
- Internet access at first startup to download the model, tokenizer, and upstream
  code; cached assets can be used afterward.

The server loads one Nemotron 3B instance on GPU 0 with bitsandbytes 4-bit NF4
quantization, double quantization, and `bfloat16` compute. It uses
`trust_remote_code=True`; the mask token ID is read from the loaded model.
The current dependency set pins `transformers==5.1.0` in `pyproject.toml` and
is recorded in `uv.lock`.
Memory use depends on the input length and model runtime. Run one server worker,
since each worker would load another model instance.

```sh
make install
make serve
curl http://127.0.0.1:8000/health
```

The default address is `127.0.0.1:8000`. API documentation is at
<http://127.0.0.1:8000/docs>. To change the port, use `make serve PORT=8080`.
The equivalent direct commands are:

```sh
uv sync --locked
uv run --locked maskdecide --host 127.0.0.1 --port 8000
```

## Local decisions: `/decide`

`POST /decide` accepts `context`, `question`, `type`, and optional `options`.

| Type | Options | Response `answer` |
| --- | --- | --- |
| `boolean` | None | Boolean |
| `single_choice` | 2–20 | One option ID |
| `multiple_choice` | 2–20 | List of selected option IDs, possibly empty |

Each option has a unique `id` and a `text` description. IDs must start with an
ASCII letter, contain only letters, digits, or underscores, and have at most 40
characters.

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

An illustrative response if the model selects `billing`:

```json
{
  "type": "single_choice",
  "answer": "billing",
  "model": "nvidia/Nemotron-Labs-Diffusion-3B"
}
```

## Experimental System One adapter: `/v1/systemone`

`POST /v1/systemone` accepts a shared `state` and a map of `questions`. State
and instructions may be text, an object, or an array. The `model` field is
accepted for interface compatibility; it does not select the backend. The
response always identifies `nvidia/Nemotron-Labs-Diffusion-3B`.

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
        "criteria": ["General inquiry", "Needs assistance", "Service-wide outage"]
      }
    }
  }'
```

Responses contain `answers` keyed by question ID and `usage` with input token
and model forward counts. `usage.output_tokens` is zero because the service
fills internal answer slots rather than generating free-form output.

| Question | Criteria | Current response meaning |
| --- | --- | --- |
| `noul` | Optional `true`/`false` descriptions | `noul` is the restricted-softmax probability of the Yes label |
| `choice` | 2–255 keyed alternatives | Selected key, distribution across all keys, and `confidence` equal to its maximum |
| `score` | 2–10 levels | Probability-weighted level index, `legend`, distribution, and maximum probability |

The `isolated` request field is accepted but ignored. The server groups ordinary
questions using its `MAX_QUESTIONS_PER_PROMPT` policy (currently 30); large
choices are evaluated separately. Compatibility is limited to the implemented
HTTP shapes, not all Jev SDK behavior or probabilistic semantics. A request may
contain at most 128 questions, subject to the token limit for each prompt.

### How large choices work

Choices with 2–14 alternatives use one masked answer slot with one token label
per option. With 15–255 alternatives, the adapter compares two candidates at a
time in both orders and advances the winner through a tournament. This counters
the model's strong position bias seen with long option lists. A tournament with
`N` alternatives needs `2 × (N - 1)` model forwards. Defeat margins give each
candidate a score; a softmax converts those scores into the required response
distribution. This distribution is **not calibrated confidence** and the
tournament can still make an incorrect decision. The public request and response
shapes are the same at either option count.

## Inference, attention, and logs

The API serializes state, questions, and rubrics with the Nemotron chat template.
The assistant turn is left open with masked answer slots. Permitted labels must
each tokenize as one token; the decoder reads the hidden state at each mask
position and projects only the permitted output rows in float32. User text that
contains the model's mask token is rejected. The default `one_pass` mode scores
all unresolved slots in one forward. Optional `iterative` mode commits selected
slots and recomputes the full bidirectional sequence; it does not use the old
Fast-dLLM block cache or preceding-token shift.

Structured attention assigns shared state to group 0 and each question and its
answer slots to a positive group. A question can directly attend to its own
group and the shared state; it cannot directly attend to another question's
group. Shared-state tokens can attend to all groups, so this is **not complete
cross-question isolation**. The server installs a Nemotron-specific attention
backend and runs a startup probe to verify it is used by the loaded model.

For each evaluated prompt, logs include `PROMPT` (rendered chat text),
`ASSISTANT INPUT` (the actual tokenized text with masks), `SLOTS` (positions,
allowed labels, selected labels, logits, and restricted probabilities), and
`ASSISTANT RESULT` (the same scaffold with labels filled in). Tournament logs
include the final winner and leading scores. `ASSISTANT RESULT` is a filled
answer scaffold, not additional prose generated by the model. These logs can
contain the full input state and question text. The new HTTP middleware currently
buffers request and response bodies, but its structured body-log emission is
commented out; the inference logs above are the active detailed logs.

## Configuration

| Setting | Current default | Purpose |
| --- | --- | --- |
| `MAX_INPUT_TOKENS` | `4096` | Maximum encoded tokens in each evaluated prompt |
| `LOCAL_API_KEY` | Unset | Optional bearer token for both POST endpoints |
| `DECODER_MODE` | `one_pass` | `one_pass` or `iterative` |
| `DECODER_THRESHOLD` | `0.85` | Iterative scheduling threshold in `(0, 1]` |
| Make `HOST` / CLI `--host` | `127.0.0.1` | Bind address |
| Make `PORT` / CLI `--port` | `8000` | Listen port |

`MAX_QUESTIONS_PER_PROMPT=30`, structured attention, and the pairwise-choice
threshold of 15 are currently constants in `src/maskdecide/api.py`, not
environment variables. Environment variables are read at startup; an `.env`
file is not loaded automatically. With `LOCAL_API_KEY` set, send
`Authorization: Bearer <your-key>` in POST requests. `/health` and `/docs`
remain public. Changing the model or decoder mode requires restarting the
server.

```sh
DECODER_MODE=iterative DECODER_THRESHOLD=0.9 make serve
```

## Tests and current results

```sh
make test
# In another terminal, while the server is running:
make test-smoke
make test-smoke TEST_ARGS='--case "Choice sampled"'
make test-smoke TEST_ARGS='--repeat 3'
```

The smoke script lives at `tests/test_jev_api_smoke.py`. It has **45 cases and
55 expected decisions** per run, including booleans, negation, structured
inputs, numeric scores, mixed questions, and choices with 15–65 alternatives.
Its sampled choice targets are reproducible. `--list` shows all cases;
repeated `--case` arguments select cases by name. The script exits nonzero if
any decision is wrong or a request fails. Its default per-request timeout is
300 seconds because large tournaments can take longer than ordinary questions.

On **2026-09-26**, the running Nemotron 3B server scored **54/55 correct
decisions with zero failed requests**. The remaining error was `Score: check
scale and distribution / range`. This small, synthetic smoke suite is a
regression aid, not a representative benchmark or proof of calibration. Large
choices are slower: the observed 65-option request took roughly 12 seconds.

The local unit suite currently needs maintenance: `make test` reaches 43 tests
but reports 9 errors in older `test_decoding.py` fixtures that expect the
Fast-dLLM model interface (`FakeModel` has no `diffusion_head`). The newer
contract and slot-attention suites pass individually (21 and 13 tests). This
test debt is tracked below; do not treat a passing smoke run as a passing unit
suite.

Optional diagnostics in `tests/verify_nemotron_tokenizer.py` check prompt and
slot layout with cached tokenizer assets; `tests/verify_nemotron_cpu.py` checks
the upstream model code with reduced random weights. The pairwise and boolean
probe scripts in `tests/` reproduce specific choice cases against a running
server. These scripts are diagnostics, not accuracy benchmarks.

Historical Fast-dLLM 1.5B/7B experiments are described in
[decoder comparison](docs/decoder-comparison.md) and
[inference review](docs/inference-review.md). Those results and the older
[architecture document](docs/architecture.md) do **not** describe the current
Nemotron runtime.

## Roadmap

Completed in the current working tree (not yet committed):

- [x] Replace the Fast-dLLM runtime with Nemotron Labs Diffusion 3B, update the
  Transformers dependency and lockfile, and load the mask ID from the checkpoint.
- [x] Move answer masks into the open assistant turn; read logits at mask
  positions and compare restricted candidate rows in float32.
- [x] Add structured per-question attention, a startup check on the loaded
  model, and server-controlled prompt grouping.
- [x] Log the rendered prompt, actual masked assistant input, slot scores,
  filled answer text, and final large-choice ranking.
- [x] Replace long categorical lists with two-order pairwise tournaments from
  15 options onward; preserve the HTTP contract and expand reproducible smoke
  cases through 65 options.
- [x] Move the smoke script into `tests/` and update `make test-smoke`.
- [x] Add HTTP request/response capture middleware and Nemotron tokenizer,
  reduced-model, and live choice diagnostic scripts.

Open work:

- [ ] Fix the remaining numeric `score` smoke case, then add varied examples
  that distinguish rubric interpretation from model error.
- [ ] Rewrite the old decoder unit fixtures for Nemotron so `make test` passes;
  remove the unused binary-ranking scaffold after equivalent coverage exists.
- [ ] Benchmark accuracy across varied tasks, option orders, and larger lists,
  including a small autoregressive baseline on the same hardware; measure
  tournament latency and evaluate fewer or batched comparisons.
- [ ] Study calibration of `noul`, `score`, and choice distributions before
  using them as confidence thresholds.
- [ ] Make request isolation effective or document and test the precise
  cross-question effects of the shared-state attention path.
- [ ] Make full prompt/body logging optional and redact sensitive fields where
  needed; review the HTTP logging middleware's current buffering behavior.
- [ ] Consolidate `README2.md` and update `docs/architecture.md`, package metadata, and
  `THIRD_PARTY_NOTICES.md` for Nemotron. They still contain Fast-dLLM-era text.

## License and acknowledgments

MaskDecide's original code and documentation are licensed under the
[Apache License 2.0](LICENSE). Model weights, tokenizer assets, upstream code,
and dependencies retain their own terms and are downloaded separately. The
current [third-party notices](THIRD_PARTY_NOTICES.md) still describe the prior
Fast-dLLM model and need updating before they can serve as a record for the
Nemotron runtime. The HTTP adapter's interface is inspired by
[Jev from TypeSafe AI](https://typesafe.ai/).
