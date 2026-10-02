# Task 2 manual error analysis

The training script creates candidates in `outputs/error_review.json`. Before final
submission, inspect the actual review text for each selected record and complete the
table below. Do not invent examples or copy the same example into multiple categories.

| Category | Required count | Completed count | What to record |
|---|---:|---:|---|
| Confident false positive | 5 | 0 | Text, probability, error type, explanation, testable fix |
| Confident false negative | 5 | 0 | Text, probability, error type, explanation, testable fix |
| Near threshold | 5 | 0 | Text, probability, error type, explanation, testable fix |
| Slice-specific | 5 | 0 | Text, length slice, error type, explanation, testable fix |

Suggested error labels include negation failure, sarcasm, mixed sentiment, aspect
conflict, unusual wording, spelling/noise, insufficient context, and long-review
context loss. Each proposed fix must be testable, such as retaining negation words,
changing the tokenizer, adding an attention-pooling head, or evaluating a different
maximum sequence length.

This file is intentionally not filled with fabricated review text. The final 20 entries
must be completed from the actual generated `outputs/error_review.json` after the run.
