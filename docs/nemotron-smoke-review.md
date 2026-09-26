# Nemotron smoke review — 2026-09-25

> Follow-up: the assistant-prefill change implemented after this review moves
> slots into the open assistant turn. Questions/rubrics remain in the user
> message and are repeated in the assistant, with both copies sharing their
> attention group. The findings below describe the earlier layout; accuracy
> effects still require a future inference run. The offline regression is
> `HF_HUB_OFFLINE=1 .venv/bin/python tests/verify_nemotron_tokenizer.py`.

This is a static and tokenizer-only review of the reported 38/49 smoke run.
No checkpoint weights were loaded, no model inference was run, and no HTTP
requests were sent. Production behavior was not changed during this review.
The summary does not identify the running server's source version or expose
per-slot logits, so it cannot establish a unique cause for the failures.

## Verified on the current working tree

All 39 smoke cases were reconstructed using the cached Nemotron 3B tokenizer
at revision `0d51902da1f8869f83413ce642fab402fa5641e0` and the real API builders.
A CPU device stub was supplied only because input assembly needs a device.
All 49 answer slots passed these checks:

- The slot ends its own question's token group, after the heading and rubric.
- Every question type, including score, has a positive group ID.
- The slot and its question can attend to each other in both directions.
- Direct attention from the slot to other positive groups is blocked.
- Every permitted answer label encodes as a single candidate token.

The rendered sequences from this audit are in
`/tmp/maskdecide-smoke-layout.txt` (a temporary local diagnostic artifact).
These checks validate layout and the constructed attention matrix, not actual
checkpoint predictions or whether a running server has loaded these changes.

The inspected cached NVIDIA forward projects `last_hidden_state` through
`diffusion_head` at the mask position. The current decoder's unshifted indexing
matches that diffusion path. The candidate-column mapping also restores each
slot's label order after projecting the union of candidate IDs.

## Findings and hypotheses

### 1. Inline slots now lie inside the user message

The earlier inline-placement change puts answer masks before the user message's
closing delimiter. The actual false-statement sequence ends approximately as:

```text
<|im_start|>user
...
Question 1 (noul): Is the result of 2 + 2 equal to 5?
A: Yes / True
B: No / False
Question 1: Is the answer yes/true? Answer (A for true or B False):<SPECIAL_100>
<|im_end|>
<|im_start|>assistant
<think></think>
```

The adjacency requirement is satisfied, but the model fills a hole in user
content rather than generating an assistant answer after the generation prefix.
This is a concrete formatting difference and a hypothesis for quality changes,
not proof that diffusion cannot fill such holes or that this caused the reported
run. NVIDIA's example instead generates a continuation after the chat prompt:
[official model card](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B).

Any future layout comparison should keep question/answer adjacency and the
attention groups while varying the role placement explicitly.

### 2. The boolean failures all have the same direction

There are 12 expected-true and 12 expected-false booleans in the suite. The nine
reported boolean failures all expect false; therefore the reported run got all
12 true cases and only 3 of 12 false cases right. In the current mapping,
`A = true`, and the smoke test classifies `p(A) >= 0.5` as true.

This supports investigating positive-label bias or weak/tied discrimination.
It does not establish a universal first-option bias: several choice tests with
the correct answer in a later position passed, and the summary does not show
which options the failing 26/27-option cases actually selected.

The boolean scaffold also repeats a generic affirmative question immediately
before each slot. Comparing it with a neutral `Answer:` label is a useful
controlled experiment; improvement has not been measured.

### 3. Ties deterministically select the first label

`restricted.argmax()` returns the first maximum. `[A, B]` with equal logits
therefore selects A; the corresponding probability is 0.5, which the smoke
test also classifies as true. A tied choice similarly selects the earliest
maximal candidate. There is no unconditional first-option fallback in the
decoder; this rule applies to tied maxima.

The projection converts its result to float32 *after* `F.linear`. If projection
inputs are bfloat16, this does not recover distinctions rounded away in its
output. A CPU illustration, not checkpoint evidence:
`[20.01, 20.02] -> bfloat16 -> float32 = [20.0, 20.0]`.
Actual per-slot logits and the projection dtype are needed before blaming ties
or quantization for these failures.

### 4. Attention groups do not provide full isolation

Shared-state/group-0 rows attend to all questions. Their representations can
carry information from question B back to question A in later layers, despite
blocked direct question-to-question edges. Template prefix/suffix tokens also
belong to group 0. The startup probe intentionally checks that changing a
question affects shared-state representations; it does not certify independence.

This can matter for the mixed-question cases but cannot explain every failure:
the arithmetic false statement, exact invoice threshold, and large choices
already have only one question. `isolated: true` is currently ignored; separate
prompts require the server's `MAX_QUESTIONS_PER_PROMPT = 1` policy.

### 5. Iterative mode would still use one forward for these Jev questions

Each Jev question currently has one mask in a distinct positive group. The
iterative scheduler accepts at least one proposal per active group, even below
the threshold. It consequently commits every Jev slot on the first forward.
Switching from one_pass to iterative alone does not add refinement here.

### 6. The 26/27-option failures are not an invalid-AA-token boundary

Both now use a single categorical answer slot. The tokenizer accepts the
space-prefixed tokens ` A=1349`, ` B=1398`, ` Z=2163`, and ` AA=34606`.
The expected answers correctly map to Z/item_26 and AA/item_27. The old
27-option binary-ranking explanation in historical documentation does not
describe the current implementation. Label familiarity, position sensitivity,
prompt wording, and uncertainty remain hypotheses requiring actual logits.

## Diagnostic gaps and next comparisons

The current `logger.info(text)` logs the prompt *before* slots are inserted.
It is not evidence of the exact token sequence passed to the model. Useful
opt-in diagnostics would capture the decoded assembled input, slot index,
question ID, group, candidate IDs, raw logits, top-two margin, and tie count.

The smoke summary's one-hot/confidence=1.0 note is unconditional and outdated
for this decoder: the API normally returns restricted-softmax distributions.
That printed sentence does not demonstrate saturated probabilities.

Before changing model behavior, capture those diagnostics on a future run.
Then compare one variable at a time: user-versus-assistant slot placement,
neutral boolean scaffold, swapped boolean labels, independent prompts, and
float32 candidate projection if actual ties are present. Preserve option and
question order variants and evaluate additional examples rather than tuning
only these 39 cases. None of these changes is a demonstrated accuracy fix yet.
