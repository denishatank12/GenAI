"""Windows/VS Code entry point for the complete Task 3 pipeline.

Run from any directory:
    python task3_gan/member_denisha/run_task3.py

The script always runs the smoke test first. Full training starts only when
the smoke test passes. It resumes from the latest checkpoint when available.
The approved course images must already be placed in task3_gan/data/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


MEMBER_DIR = Path(__file__).resolve().parent
REPO_ROOT = MEMBER_DIR.parents[1]
DATA_ROOT = MEMBER_DIR.parent / "data"
CHECKPOINT_DIR = MEMBER_DIR / "checkpoints"
LOG_DIR = REPO_ROOT / "reproducibility" / "raw_logs"
LOG_PATH = LOG_DIR / "task3_denisha.log"
CONFIG = MEMBER_DIR / "config.yaml"


def run(command: list[str], *, cwd: Path) -> None:
    print("\n$", " ".join(str(x) for x in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def verify_data() -> None:
    expected = {
        "monet_jpg": DATA_ROOT / "monet_jpg",
        "photo_jpg": DATA_ROOT / "photo_jpg",
    }
    missing = []
    for name, folder in expected.items():
        images = [p for p in folder.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}] if folder.exists() else []
        print(f"{name}: {len(images)} images — {folder}")
        if not images:
            missing.append(str(folder))
    if missing:
        raise SystemExit(
            "Approved image data is missing. Add the course-provided Kaggle data to:\n"
            f"  {DATA_ROOT / 'monet_jpg'}\n"
            f"  {DATA_ROOT / 'photo_jpg'}\n"
            "Do not use manually edited or externally sourced generated images."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="refuse to resume and require an empty checkpoint folder")
    args = parser.parse_args()
    config = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    verify_data()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and any(CHECKPOINT_DIR.glob("*.pt")):
        raise SystemExit(f"Checkpoint folder is not empty: {CHECKPOINT_DIR}. Use another folder or remove it deliberately.")
    run([sys.executable, "src/smoke_test.py", "--config", str(config)], cwd=MEMBER_DIR)
    if args.smoke_only:
        print("Smoke test passed; full training was intentionally skipped.")
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_command = [
        sys.executable, "-u", "src/train.py", "--config", str(config),
        "--checkpoint-dir", str(CHECKPOINT_DIR), "--resume-latest",
    ]
    print(f"\nTraining log: {LOG_PATH}")
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(train_command, cwd=MEMBER_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise SystemExit(f"Training failed with exit code {code}; evaluation was not started.")
    run([sys.executable, "src/plot_history.py"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_metrics.py", "--real", "../data/monet_jpg", "--fake", "outputs/pred_B2A", "--out", "outputs/metrics_B2A.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_metrics.py", "--real", "../data/photo_jpg", "--fake", "outputs/pred_A2B", "--out", "outputs/metrics_A2B.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_lpips.py", "--real", "../data/monet_jpg", "--recon", "outputs/recon_A", "--out", "outputs/lpips_A.json"], cwd=MEMBER_DIR)
    run([sys.executable, "src/evaluate_lpips.py", "--real", "../data/photo_jpg", "--recon", "outputs/recon_B", "--out", "outputs/lpips_B.json"], cwd=MEMBER_DIR)
    print("\nTRAINING AND AUTOMATED EVALUATION COMPLETE.")
    print("Next required step: two human raters must fill outputs/human_audit.csv before audit_agreement.py and final validation.")


if __name__ == "__main__":
    main()
