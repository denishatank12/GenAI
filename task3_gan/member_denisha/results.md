# Task 3 results - Denisha Ketan Tank

Complete this after the GPU run using `outputs/metrics.json`, `outputs/history.json`,
generated images, `human_audit.csv`, and the raw log. Report both directions for FID,
KID, generative precision/recall, cycle L1, LPIPS, content cosine similarity, loss
curves, cycle/identity losses, gradient stability, parameter count, training time,
images/sec, peak memory, and the Kaggle score/rank.

The human audit must cover 30 fixed samples with two raters and report Cohen's kappa
or percent agreement. The Kaggle submission must be generated directly from this
checkpoint's inference output.
# Task 3 results - Denisha Ketan Tank

Run the complete local Windows/VS Code pipeline with:

```text
python task3_gan/member_denisha/run_task3.py
```

The command runs the smoke test first and starts full training only if it passes. It resumes from `member_denisha/checkpoints/cyclegan_latest.pt` after an interruption. The approved Monet/photo data must be placed in `task3_gan/data/monet_jpg/` and `task3_gan/data/photo_jpg/`.
