"""Create a machine-readable run manifest without modifying raw logs."""
import argparse, importlib.metadata as md, json, platform, subprocess, sys
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument('--task', required=True); ap.add_argument('--command', required=True); ap.add_argument('--dataset', required=True); ap.add_argument('--checkpoint', action='append', default=[]); ap.add_argument('--out', required=True); args = ap.parse_args()
packages = {}
for name in ['numpy','pandas','scikit-learn','torch','torchvision','Pillow','scipy','PyYAML','lpips','torch-fidelity']:
    try: packages[name] = md.version(name)
    except md.PackageNotFoundError: pass
gpu = 'CPU only'
try:
    import torch
    if torch.cuda.is_available(): gpu = torch.cuda.get_device_name(0)
except Exception: pass
manifest = {'student':'Denisha Ketan Tank','task':args.task,'command':args.command,'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'dataset_name_and_version':args.dataset,'dataset_path':args.dataset,'python_version':sys.version,'package_versions':packages,'hardware':{'platform':platform.platform(),'gpu':gpu},'checkpoint_to_result_mapping':{Path(x).name:x for x in args.checkpoint}}
Path(args.out).write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
