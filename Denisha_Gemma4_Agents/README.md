# Denisha - Gemma 4 Developer Agent variants

This folder contains two complete variants for Denisha's assigned area, tools
and integration. Both use the current competition-required base model
`gemma-4-31b-it-qat-w4a16-ct`.

| Variant | Main idea |
|---|---|
| `agent_1_tool_policy` | Source-first repository reconnaissance and conservative tool selection |
| `agent_2_integrated` | Plan -> inspect -> edit -> test -> repair with a read-only analyzer |

Each variant is packaged separately as `submission.zip` with `agent.yaml` at
the archive root. The RTX 4090 is suitable for local configuration testing,
but Kaggle's official harness supplies the model-serving and evaluation
environment.

```bash
cd Gemma4_Denisha
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python scripts/build_submission.py --variant agent_1_tool_policy
python scripts/build_submission.py --variant agent_2_integrated
```

Record local and Kaggle scores, runtime, tool calls, GPU, changes, and lessons
in `experiments/experiment_log.csv`. Keep at least one failed experiment.
