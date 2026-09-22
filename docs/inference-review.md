# Inference review — September 21, 2026

This review records the original single-pass implementation. A constrained
iterative decoder was implemented afterward; see the
[decoder comparison](decoder-comparison.md) for its design and measured results.

## Conclusion

No incorrect mask ID, logit offset, option index, or answer mapping was found in
the reviewed implementation. The smoke failures are real, but the experiments
do not support attributing all of them to a simple implementation bug.

The current inference procedure differs from the upstream decoder, and its
prompting and large-choice ranking introduce additional sources of error. These
are algorithmic limitations to investigate, not demonstrated fixes. The server's
inference code was left unchanged after the review.

## Checks against the loaded model

Model: `Efficient-Large-Model/Fast_dLLM_v2_1.5B`, cached implementation revision
`25093b6f63300adfd57f72145083c8a528fe4f16`, loaded in bfloat16 on an NVIDIA
GeForce RTX 4060 Laptop GPU.

- Token `151665` decodes to `|<MASK>|` in the actual tokenizer.
- Both `A` and ` A` are valid single tokens, with different IDs (`32` and `362`).
  Restricting the candidates to space-prefixed labels is a choice of output
  representation, not an invalid-token error.
- Reading logits at `mask_position - 1` follows Fast-dLLM v2's token shift.
  The upstream decoder shifts logits before selecting masked positions, and
  separately handles the first token of the next block.
- Slot indices and mappings to boolean, choice, and score answers agree with
  the selected labels. In failing score cases, the model actually ranked the
  permitted ` B` token highest; the adapter did not accidentally map another
  label to the middle level.
- The model's evaluation attention is block-causal with a block size of 32:
  a position can attend to its own block and earlier blocks. The model constructs
  this mask internally, replacing a supplied attention mask in this path.

Reference: [Fast-dLLM v2 architecture and decoding](https://github.com/NVlabs/Fast-dLLM/tree/main/v2).
The cached `modeling.py` was inspected directly, including `eval_block_diff_mask`,
the model `forward`, and `generate`.

## Controlled experiments

Each variant used all 39 existing smoke cases (49 expected decisions), unchanged
expected answers, the same model, and greedy label selection. Except where stated,
each variant changed only the indicated part of the baseline. These are single
diagnostic runs on a small suite, not general accuracy estimates or benchmarks.

| Variant | Correct decisions |
| --- | --- |
| Current implementation | 37/49 |
| Score bare labels (`A`) instead of space-prefixed labels (` A`) | 38/49 |
| Combine the logits of both label spellings with log-sum-exp | 37/49 |
| Set attention block size to 1 | 39/49 |
| Append masks to complete the last 32-token block | 36/49 |
| Shorten answer scaffolding to `Question N:` | 35/49 |
| Fill slots sequentially, rerunning the full input after each selection | 38/49 |
| Evaluate each question in its own prompt | 38/49 |
| Repeat the original instructions immediately before each answer slot | 38/49 |

Sequential slot filling is a diagnostic ablation, not a reproduction of the
upstream generation algorithm. Changing block size to 1 changes the model's
attention pattern; its small improvement is not evidence that block size 32 is
incorrect. No variant was adopted on the basis of this suite.

## Findings

### Multiple masked answers are not isolated

All question rubrics share one prompt, and later slots can attend to earlier
masked slots. The upstream generation loop completes a block before advancing;
MaskDecide evaluates all slots in one pass, including slots in later blocks
whose preceding blocks still contain unresolved masks. This is a departure from
the upstream inference procedure, rather than an off-by-one error.

There is direct evidence of question interaction: reversing the Hanna/Jeff
questions changes Jeff's answer. Evaluating questions independently or filling
slots sequentially fixes the second answer in the Maya and historical outage
cases, but also produces regressions elsewhere. Neither change resolves all
failures. Isolation also changes the prompt length, question numbering, and
block alignment, so these experiments do not identify a single causal mechanism.

### Large choices use a different decision problem

The 26-option case selects the highest-scoring label at a single position and
passes. The 27-option case ranks 27 separate yes/no logit margins and fails.
In this run the baseline selected item 20, sequential filling selected item 19,
and the expected answer was item 27.

Those margins come from different positions and conditioning contexts. They are
not one joint categorical distribution over the options. The ranking is an
experimental heuristic; correct indexing alone cannot make it equivalent to
the small-choice method. One case on each side of the boundary is insufficient
to quantify the effect.

### Some failures persist for a single question

The Hanna addressee question also fails when asked alone. The numeric scale
selects the middle label for 0, 3, 8, and 10 in the baseline even though the
rubric assigns those numbers to the low or high level. Those four score failures
persist under both sequential filling and per-question evaluation.

For the explicit overdue-invoice rubric, 12 days and 10 days both select false.
The recorded baseline logits for the allowed ` A` and ` B` tokens are identical
at the model's bfloat16 output precision: `[22.375, 22.875]`. The adapter correctly
selects ` B` in both cases. This locates the error before response construction;
it does not prove whether better prompting or another model would resolve it.

## Next investigations

1. Compare answer scaffolds and instruction formats on a separate evaluation
   set, including newly generated numeric and temporal cases. Avoid tuning only
   against the current smoke suite.
2. Evaluate an explicit isolation mode or batched independent prompts, measuring
   accuracy, memory, and latency together. This changes the current shared-prompt
   design and needs a deliberate choice.
3. Rework or limit large-choice ranking only after testing more option counts,
   answer positions, and order permutations.
4. Keep confidence calibration separate: one-hot response probabilities and
   `confidence=1.0` do not cause the argmax errors, but cannot reveal uncertainty.
