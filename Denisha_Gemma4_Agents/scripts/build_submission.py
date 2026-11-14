from __future__ import annotations
import argparse, shutil, zipfile
from pathlib import Path
from smoke_test import ROOT, check_variant

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--variant',required=True,choices=['agent_1_tool_policy','agent_2_integrated']); ap.add_argument('--output',default='build'); a=ap.parse_args()
    src=ROOT/a.variant; check_variant(src); out=ROOT/a.output/a.variant; shutil.rmtree(out,ignore_errors=True); out.mkdir(parents=True)
    for item in src.iterdir(): shutil.copytree(item,out/item.name) if item.is_dir() else shutil.copy2(item,out/item.name)
    archive=ROOT/a.output/f'{a.variant}.submission.zip'; archive.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file(): z.write(p,p.relative_to(out))
    print(archive)
if __name__=='__main__': main()
