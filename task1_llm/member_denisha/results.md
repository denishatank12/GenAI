# Task 1 results - Denisha Ketan Tank

This file must be completed from the raw run outputs after training. Do not invent
metrics. The authoritative files will be `outputs/metrics.csv`, `outputs/history.json`,
`outputs/sample.txt`, and the untouched raw log in `reproducibility/raw_logs/`.

## Architecture and justification

Describe the tokenizer, sequence length, embedding size, number of heads/layers,
causal mask, optimizer, warm-up schedule, and why these choices were made.

## Metrics

Copy the required values from `outputs/metrics.csv`: train/validation cross-entropy,
perplexity, bits-per-character, generalization gap, top-1 accuracy, Distinct-1/2/3,
repeated 4-gram rate, parameter count, tokens/sec, generation tokens/sec, peak memory,
and total training time.

## Failure analysis

Attach three generated snippets from `outputs/sample.txt`. For each, identify the
failure category and a testable improvement. This analysis must be written and
understood by Denisha for the viva.
