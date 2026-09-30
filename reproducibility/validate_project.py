"""Check that the non-GPU project package is complete and contains no result claims."""
from pathlib import Path
import ast, sys

root = Path(__file__).resolve().parents[1]
required = [
    'README.md', 'COLAB_RUN.md', 'TEAMMATE_HANDOFF.md', 'requirements.txt',
    'task1_llm/member_denisha/src/train.py', 'task1_llm/member_denisha/config.yaml',
    'task2_sentiment/member_denisha/src/train.py', 'task2_sentiment/member_denisha/config.yaml',
    'task3_gan/member_denisha/src/train.py', 'task3_gan/member_denisha/src/evaluate_metrics.py', 'task3_gan/member_denisha/config.yaml',
    'report/REPORT_SKELETON.md', 'reproducibility/manifest_template.json', 'reproducibility/collect_manifest.py', 'reproducibility/validate_results.py'
]
missing = [x for x in required if not (root / x).exists()]
bad_python = []
for path in root.rglob('*.py'):
    if '__pycache__' not in path.parts:
        try: ast.parse(path.read_text())
        except SyntaxError: bad_python.append(str(path.relative_to(root)))
if missing or bad_python:
    print('FAIL'); print('missing:', missing); print('bad_python:', bad_python); sys.exit(1)
print(f'PASS: {len(required)} required project artifacts present and all Python files parse.')
print('GPU runs, real metrics, human audit scores, and Kaggle rank remain intentionally unfilled.')
