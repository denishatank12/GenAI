from __future__ import annotations
import argparse
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'gemma-4-31b-it-qat-w4a16-ct'
ALLOWED = {'run_command','read_file','edit_file','write_file','get_status','submit_patch','get_code_neighbors','search_similar_code','get_code_subgraph','agent_tool'}

class KaggleLoader(yaml.SafeLoader):
    pass

KaggleLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))

def check_includes(text: str, base: Path, sandbox_root: Path):
    for line in text.splitlines():
        if '!include' not in line: continue
        rel=line.split('!include',1)[1].strip().strip('"\'')
        target=(base / rel).resolve()
        if not target.is_relative_to(sandbox_root.resolve()): raise AssertionError(f'unsafe include: {rel}')
        if not target.exists(): raise AssertionError(f'missing include: {target}')

def check_variant(path: Path):
    root=path/'agent.yaml'; data=yaml.load(root.read_text(), Loader=KaggleLoader)
    assert data['model']==MODEL, f'{path}: wrong model'
    assert data.get('name'), f'{path}: missing name'
    for item in data.get('tools',[]):
        if isinstance(item,str): assert item in ALLOWED, f'{path}: unsupported tool {item}'
        else: assert 'agent_tool' in item
    check_includes(root.read_text(), path, path)
    for p in path.rglob('*.yaml'):
        yaml.load(p.read_text(), Loader=KaggleLoader); check_includes(p.read_text(), p.parent, path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variant',choices=['agent_1_tool_policy','agent_2_integrated']); a=ap.parse_args()
    variants=[ROOT/a.variant] if a.variant else [ROOT/'agent_1_tool_policy',ROOT/'agent_2_integrated']
    for v in variants: check_variant(v); print('PASS:',v.name)
    print('All Gemma agent configuration smoke tests passed')
if __name__=='__main__': main()
