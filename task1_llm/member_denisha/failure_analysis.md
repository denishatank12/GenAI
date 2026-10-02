# Task 1 sequence-model failure analysis

Complete this analysis from the actual generated text in `outputs/sample.txt` after the
GPU run. Include three distinct snippets and explain the failure in your own words.

| Case | Generated snippet | Failure type | Observation | Testable improvement |
|---|---|---|---|---|
| 1 | Paste an actual snippet here. | Repetition / other | Explain what repeated or failed. | Change temperature, context length, or training schedule and compare repetition rate. |
| 2 | Paste an actual snippet here. | Grammar / coherence / other | Explain the specific failure. | Test a longer training run or a different decoding temperature. |
| 3 | Paste an actual snippet here. | Hallucination / other | Explain why the continuation fails. | Compare sampling temperatures or add more training data. |

Do not invent snippets. The final three cases must be taken from the untouched GPU-run
`outputs/sample.txt` and linked to the corresponding checkpoint and metrics file.
