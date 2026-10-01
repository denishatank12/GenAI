# Task 1 training-workload fix

The original dataset class used a stride of one character implicitly:

```text
number of examples = number of characters - sequence_length - 1
```

For approximately 88.4M training characters and sequence length 128, that produced
approximately 88.4M overlapping examples, 1.38M batches per epoch at batch size 64,
and roughly 13.8M batches across ten epochs. The process was active, but it could
reasonably appear frozen because the first print occurred only after training and
validation completed.

The lab specifies fixed-length input-target sequences but does not require a
stride-one window. The corrected implementation exposes `stride` in the config and
defaults it to `sequence_length` (non-overlapping chunks). It preserves the required
100,000-story training split, 10,000-story validation split, sequence length, and ten
epochs. With the current competition config (`stride: 128`, `batch_size: 48`), the
expected workload is approximately 691K training sequences, 14.4K batches per
epoch, and 144K batches total. The smaller batch and tuned `learning_rate: 0.0005`
match the configuration that produced the observed 78% validation accuracy and
0.65 loss; the exact result depends on the dataset split and hardware.

Additional reliability changes:

- Prints dataset sizes and batch counts before training.
- Prints progress every configured number of batches.
- Streams JSONL loading and prints loading progress every 10,000 stories.
- Writes `outputs/history.json` after every epoch.
- Saves an epoch checkpoint after every epoch, so an interrupted run retains progress.
