# GPU Lab / Colab run guide

Run each task from its `member_denisha` directory after installing the requirements.
Use a fresh GPU runtime for Tasks 1 and 3. Keep the terminal output as the raw log;
do not edit it after the run.

```bash
git clone YOUR_REPO_URL
cd DATA266_Lab1_Denisha
pip install -r requirements.txt
```

The two Hugging Face datasets can be exported with:

```bash
python reproducibility/prepare_data.py --task task1 --out .
python reproducibility/prepare_data.py --task task2 --out .
```

Use the course-approved Kaggle source for the CycleGAN images; do not put Kaggle
credentials in the repository.

## Task 1

Place the approved TinyStories JSONL file at:
`task1_llm/data/TinyStories-train.jsonl`

```bash
cd task1_llm/member_denisha
python src/train.py --config config.yaml 2>&1 | tee ../../../reproducibility/raw_logs/task1_denisha.log
```

Expected outputs include `outputs/metrics.csv`, `outputs/history.json`,
`outputs/sample.txt`, `outputs/vocab.json`, and
`checkpoints/gpt_from_scratch.pt`.

## Task 2

Place the approved Yelp Polarity CSV at:
`task2_sentiment/data/yelp_polarity.csv`

```bash
cd task2_sentiment/member_denisha
python src/train.py --config config.yaml 2>&1 | tee ../../../reproducibility/raw_logs/task2_denisha.log
```

Expected outputs include three checkpoints, `outputs/metrics.json`,
`outputs/metrics_summary.csv`, and `outputs/error_review.json`.

## Task 3

Place the approved unpaired images at:

```text
task3_gan/data/monet_jpg/
task3_gan/data/photo_jpg/
```

```bash
cd task3_gan/member_denisha
python src/train.py --config config.yaml 2>&1 | tee ../../../reproducibility/raw_logs/task3_denisha.log
```

Expected outputs include CycleGAN checkpoints, loss history, generated images in both
directions, `outputs/metrics.json`, and `outputs/human_audit.csv`.

After training, run the evaluation scripts for each direction and for both cycle
reconstruction folders:

```bash
python src/evaluate_metrics.py --real ../data/monet_jpg --fake outputs/pred_B2A --out outputs/metrics_B2A.json
python src/evaluate_metrics.py --real ../data/photo_jpg --fake outputs/pred_A2B --out outputs/metrics_A2B.json
python src/evaluate_lpips.py --real ../data/monet_jpg --recon outputs/recon_A --out outputs/lpips_A.json
python src/evaluate_lpips.py --real ../data/photo_jpg --recon outputs/recon_B --out outputs/lpips_B.json
```

Fill both-rater scores into `outputs/human_audit.csv`, then run
`python src/audit_agreement.py`.

After each run, record Python/package versions, GPU model, peak memory, command, and
checkpoint mapping in a copy of `reproducibility/manifest_template.json`.
