"""Prepare the approved datasets without embedding credentials in the repository.

TinyStories and Yelp can be exported through Hugging Face `datasets`. CycleGAN's
Monet/photo data should be downloaded through the course-approved Kaggle source and
then placed in the paths documented in COLAB_RUN.md.
"""
import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--out', default='.'); ap.add_argument('--task', choices=['task1','task2','instructions'], default='instructions'); args = ap.parse_args(); out = Path(args.out)
if args.task == 'instructions':
    print('Task 1: export roneneldan/TinyStories to task1_llm/data/TinyStories-train.jsonl')
    print('Task 2: export fancyzhx/yelp_polarity to task2_sentiment/data/yelp_polarity.csv')
    print('Task 3: use the course-approved Kaggle download; place Monet and photo folders under task3_gan/data/')
elif args.task == 'task1':
    from datasets import load_dataset
    target = out / 'task1_llm' / 'data'; target.mkdir(parents=True, exist_ok=True); ds = load_dataset('roneneldan/TinyStories', split='train'); ds.to_json(str(target / 'TinyStories-train.jsonl'))
elif args.task == 'task2':
    from datasets import load_dataset
    target = out / 'task2_sentiment' / 'data'; target.mkdir(parents=True, exist_ok=True); ds = load_dataset('fancyzhx/yelp_polarity', split='train'); ds.to_csv(str(target / 'yelp_polarity.csv'))
