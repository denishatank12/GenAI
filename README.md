# DATA266 Lab 1 - Denisha Ketan Tank

This repository contains Denisha Tank's independent work for all three Lab 1 tasks:

1. A GPT-style character-level language model implemented from scratch.
2. Yelp Polarity sentiment classification with three independently trained models.
3. A CycleGAN trained on unpaired Monet and photo domains.

## Reproducibility

The canonical environment is defined in `requirements.txt`. Each run should be made
from a task config, with the command, git revision, hardware, package versions, and
raw stdout/stderr saved under `reproducibility/`. Do not edit raw logs after a run.

The GPU-intensive training commands are intended for Google Colab or the DATA266 GPU
Lab. Replace the placeholder dataset paths in the task configs with the approved raw
datasets before training.

## Current status

Task 1 now includes Denisha's completed Tesla T4 run, final metrics, loss curve,
generated samples, failure analysis, raw log, manifest, and final/resumable
checkpoints. Task 2 and Task 3 retain their own run-specific evidence; any
remaining human-audit or Kaggle fields must be completed from the final team run.

## Layout

Each task contains Denisha's code, checkpoints, outputs, processed data, metrics, and
analysis. The team member's work must remain independent; shared evaluation scripts
and comparison tables belong in the team-level report only.

Task 1's complete evidence map and checklist are in
`task1_llm/member_denisha/SUBMISSION_CHECKLIST.md`.

## Academic integrity

All architecture choices, metric interpretation, and failure analysis must be reviewed
and understood by Denisha before submission. This repository records experiments and
evidence; it is not a substitute for the individual viva.
