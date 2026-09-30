"""Validate real post-GPU outputs before they are copied into the final report."""
import argparse, json, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--task', choices=['task1','task2','task3'], required=True); ap.add_argument('--root', default='.'); args = ap.parse_args(); root = Path(args.root)
errors = []
if args.task == 'task1':
    out, ck = root/'task1_llm/member_denisha/outputs', root/'task1_llm/member_denisha/checkpoints'
    for p in [out/'metrics.csv', out/'history.json', out/'samples.json', out/'sample.txt', ck/'gpt_from_scratch.pt']:
        if not p.exists(): errors.append(f'missing {p}')
if args.task == 'task2':
    out, ck = root/'task2_sentiment/member_denisha/outputs', root/'task2_sentiment/member_denisha/checkpoints'
    for p in [out/'metrics.json', out/'metrics_summary.csv', out/'error_review.json', ck/'baseline.pt', ck/'experimental_cnn.pt', ck/'experimental_gru.pt']:
        if not p.exists(): errors.append(f'missing {p}')
    if (out/'metrics.json').exists():
        data = json.loads((out/'metrics.json').read_text());
        if len(data) != 3: errors.append('metrics.json must contain baseline and two experimental models')
    if (out/'error_review.json').exists():
        rows = json.loads((out/'error_review.json').read_text()); buckets = {r.get('review_bucket') for r in rows}
        for b in ['confident_false_positive','confident_false_negative','near_threshold','slice_specific']:
            if b not in buckets: errors.append(f'error review missing bucket {b}')
if args.task == 'task3':
    out, ck = root/'task3_gan/member_denisha/outputs', root/'task3_gan/member_denisha/checkpoints'
    for p in [out/'metrics.json', out/'history.json', out/'human_audit.csv']:
        if not p.exists(): errors.append(f'missing {p}')
    for d in [out/'pred_A2B', out/'pred_B2A', out/'recon_A', out/'recon_B']:
        if not d.exists() or len(list(d.glob('*'))) == 0: errors.append(f'missing generated images in {d}')
    if (out/'human_audit.csv').exists():
        import pandas as pd
        df = pd.read_csv(out/'human_audit.csv')
        if len(df) < 30: errors.append('human audit must contain at least 30 fixed samples')
        if any(df[c].astype(str).str.strip().eq('').any() for c in ['style_score_1','style_score_2','content_score_1','content_score_2','artifact_score_1','artifact_score_2']): errors.append('human audit has blank rater scores')
if errors:
    print('NOT READY'); print('\n'.join(f'- {e}' for e in errors)); sys.exit(1)
print(f'PASS: {args.task} has the required post-run artifacts.')
