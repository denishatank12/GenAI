import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument('--data', default='../data/yelp_polarity.csv'); ap.add_argument('--outdir', default='outputs'); args = ap.parse_args()
df = pd.read_csv(args.data); text_col = next(c for c in ['text', 'review', 'content'] if c in df); label_col = next(c for c in ['label', 'sentiment', 'target'] if c in df); df = df[[text_col, label_col]].dropna(); df['word_count'] = df[text_col].astype(str).str.split().str.len(); out = Path(args.outdir); out.mkdir(exist_ok=True)
summary = {'n_rows': int(len(df)), 'class_counts': {str(k): int(v) for k, v in df[label_col].value_counts().sort_index().items()}, 'length_mean': float(df.word_count.mean()), 'length_median': float(df.word_count.median()), 'length_p95': float(df.word_count.quantile(.95)), 'empty_or_missing_removed': int(df[text_col].astype(str).str.strip().eq('').sum())}
(out / 'data_analysis.json').write_text(json.dumps(summary, indent=2)); df.groupby(label_col).word_count.describe().to_csv(out / 'length_by_class.csv'); plt.hist(df.word_count.clip(upper=df.word_count.quantile(.99)), bins=50); plt.xlabel('Review length (words)'); plt.ylabel('Count'); plt.tight_layout(); plt.savefig(out / 'review_length_distribution.png', dpi=160)
