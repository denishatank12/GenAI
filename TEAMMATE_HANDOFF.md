# Message for Denisha Ketan Tank's teammate

The PDF requires both team members to independently complete all three tasks. We
cannot divide the lab by assigning one person Task 1 and the other person Tasks 2-3.

## Denisha's individual work

Denisha owns one complete model pipeline for each task:

- Task 1: a character-level GPT-style language model with self-attention, causal
  masking, layer normalization, feed-forward layers, residual connections, warm-up,
  scheduling, text generation, and failure analysis.
- Task 2: three independently trained Yelp Polarity classifiers: one baseline and two
  experimental models, with the full metric list and manual review of 20 errors.
- Task 3: one CycleGAN with two generators and two discriminators, both translation
  directions, full metrics, human audit, and a Kaggle submission generated directly
  from the trained model.

## Coordination we should agree on before training

1. Use the same approved evaluation scripts and dataset versions so results are
   comparable.
2. Choose different architectures or hyperparameters so the two members' models are
   genuinely independent and not near-identical.
3. Record exact hardware, package versions, commands, checkpoints, and unedited raw
   logs for every run.
4. Keep each person's code, checkpoints, outputs, metrics, and `results.md` in their
   own member folder.
5. After both individual pipelines are complete, write one report containing side-by-
   side comparison tables and joint analysis. Do not average or overwrite individual
   results.

## Current repository

Denisha's working repository is `DATA266_Lab1_Denisha/`. The first Task 1 config and
from-scratch implementation are in:

`task1_llm/member_denisha/config.yaml`

`task1_llm/member_denisha/src/train.py`

The implementation deliberately defines its own attention, causal mask, transformer
block, and generation loop; it does not call a prebuilt Transformer or attention
module. Training needs Colab or the DATA266 GPU Lab because the local environment is
not currently equipped with PyTorch/CUDA.
