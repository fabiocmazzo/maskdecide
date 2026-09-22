# Iterative decoder comparison

## Design

The iterative decoder resolves 32-token answer blocks in order. It caches only
clean, complete preceding blocks and reruns only the active block against that
cache. Candidate tokens remain restricted to the slot's allowed labels.

Within a block, predictions with restricted softmax probability at least 0.9
are committed together. If none reaches that threshold, the strongest remaining
prediction is committed. Remaining masks are then evaluated again. Every round
resolves at least one slot; the maximum number of model calls, including
prefills, is twice the slot count. The threshold is a scheduling parameter, not
calibrated confidence. The same restricted softmax now supplies the API's
`probabilities` and `confidence` fields.

The first token of a block uses the previous block's final logits. Once a slot
is committed, its raw scores are retained from the pre-commit evaluation rather
than recalculated with its answer visible. This matters for the existing
large-choice ranking heuristic.

This replaces the earlier experimental approach of rerunning the entire input
after each answer. It preserves context and scaffolding tokens and does not
perform free-form reasoning, generate extra tokens, or revise committed answers.
It is not an exact implementation of the upstream text-generation sampler.

## Methodology

Measured on September 21, 2026, with:

- NVIDIA GeForce RTX 4060 Laptop GPU.
- Fast-dLLM v2 1.5B revision `25093b6f63300adfd57f72145083c8a528fe4f16`.
- bfloat16 weights and PyTorch 2.14.0+cu130.
- 39 smoke requests / 49 expected decisions, plus 11 additional requests /
  13 expected decisions (new numeric ranges, status changes, and option positions).
- Three measured repetitions per request and mode, following one unmeasured
  warmup for that input/mode. The model stayed loaded throughout.
- Identical prompts, candidate labels, expected answers, and model weights
  between modes. The default scheduling threshold was fixed at 0.9 before
  running the comparison; it was not tuned to maximize these results.

Times cover decoder execution, including its prefills, GPU synchronization and
label selection. They exclude model loading, tokenization, prompt construction,
HTTP, and request queuing. The mode order was fixed, not randomized. Repeats
are repeatability checks, not additional independent examples.

## Results

All three repeats produced the same decisions within each mode.

| Suite | Decoder | Correct per run | Median decode time | Mean decode time |
| --- | --- | --- | --- | --- |
| Smoke | `one_pass` | 37/49 | 30.1 ms | 40.5 ms |
| Smoke | `iterative` | 40/49 | 46.6 ms | 81.7 ms |
| Additional | `one_pass` | 8/13 | 45.0 ms | 84.0 ms |
| Additional | `iterative` | 8/13 | 79.2 ms | 244.3 ms |

The iterative decoder fixed three smoke decisions: Hanna as the addressee in
the original two-question order, Maya updating the billing address, and the
service being offline at the earlier timestamp. No previously correct decision
became incorrect in these runs. It did not resolve the remaining score-boundary,
boolean-rubric, addressee, or large-choice failures.

The 27-option smoke case still failed: `one_pass` chose item 20 and `iterative`
chose item 19, while the expected answer was item 27. Its mean decoder time
increased from 247.6 ms to 1098.9 ms, with 46 model calls in iterative mode.
The large-choice scoring heuristic remains an independent limitation.

This is a small diagnostic comparison, not proof of general accuracy improvement
or a throughput benchmark. Iteration helps some interacting answer slots but
cannot supply missing model capabilities. The original decoder remains available
for workloads where its lower latency is preferable.

## Reproduce

Stop the API server first so the comparison does not load another copy of the
model into GPU memory. From the repository root:

```sh
uv run --locked python scripts/compare_decoders.py \
  --repeat 3 --output /tmp/maskdecide-decoder-comparison.json
```

The JSON report contains per-request predictions, expected answers, token counts,
model call counts, timings, and model/hardware metadata. The program finishes
successfully when measurements complete; it does not require every model answer
to be correct. `make test-smoke` remains the check that fails on wrong decisions.

Restart the API with either mode:

```sh
DECODER_MODE=iterative make serve
# Or: DECODER_MODE=one_pass make serve
```
