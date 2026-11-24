from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def require_path(label: str, path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Missing {label}: {path}")


def validate_submission(path: Path) -> None:
    require_path("submission root", path)
    root = path / "agent.yaml"
    require_path("agent.yaml", root)
    for relative in ("eval_config.yaml", "configs", "prompts", "sub_agents"):
        require_path(relative, path / relative)
    for adapter_dir in (path / "adapters").glob("*") if (path / "adapters").is_dir() else []:
        if not adapter_dir.is_dir():
            continue
        require_path(f"{adapter_dir.name}/adapter_config.json", adapter_dir / "adapter_config.json")
        require_path(f"{adapter_dir.name}/adapter_model.safetensors", adapter_dir / "adapter_model.safetensors")
        json.loads((adapter_dir / "adapter_config.json").read_text())


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the published Gemma harness locally.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--snapshots-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, default=Path("results/denisha_run"))
    parser.add_argument("--sandbox", choices=("docker", "subprocess"), default="docker")
    parser.add_argument("--max-tool-calls", type=int, default=50)
    parser.add_argument("--max-time-minutes", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()

    for label, path in (
        ("tasks file", args.tasks),
        ("snapshots directory", args.snapshots_dir),
        ("submission directory", args.submission_dir),
    ):
        require_path(label, path)

    validate_submission(args.submission_dir)
    root_configs = [p for p in args.submission_dir.glob("*.y*ml") if p.name in {"agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"}]
    if len(root_configs) != 1:
        raise SystemExit(f"Submission must contain exactly one root agent config; found {root_configs}")

    if not shutil.which("nvidia-smi"):
        raise SystemExit("nvidia-smi was not found. Connect the RTX 4090 driver before running GPU evaluation.")
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True, capture_output=True, check=False)
    if gpu.returncode != 0 or not gpu.stdout.strip():
        raise SystemExit("nvidia-smi could not see a GPU. Fix the NVIDIA driver/runtime first.")
    print(gpu.stdout.strip())

    if args.sandbox == "docker":
        if not shutil.which("docker"):
            raise SystemExit("Docker is required for --sandbox docker but was not found on PATH.")
        docker = subprocess.run(["docker", "info"], text=True, capture_output=True, check=False)
        if docker.returncode != 0:
            raise SystemExit("Docker is installed but its daemon is not running. Start Docker Desktop and retry.")

    harness = shutil.which("swegemma")
    if harness is None:
        raise SystemExit("swegemma is not installed or is not on PATH. Install the supplied harness wheel first.")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    command = [
        harness, "eval",
        "--tasks", str(args.tasks),
        "--snapshots-dir", str(args.snapshots_dir),
        "--submission-dir", str(args.submission_dir),
        "--results-dir", str(args.results_dir),
        "--sandbox", args.sandbox,
        "--max-tool-calls", str(args.max_tool_calls),
        "--max-time-minutes", str(args.max_time_minutes),
        "--concurrency", str(args.concurrency),
    ]
    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        return completed.returncode

    summary = args.results_dir / "summary.json"
    print(f"Evaluation finished. Results: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
