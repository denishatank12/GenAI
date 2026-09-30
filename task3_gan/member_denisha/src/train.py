"""Train a compact CycleGAN from scratch on two unpaired image folders.

Expected folders: data/monet_jpg and data/photo_jpg (or monet and photo).
Run from this directory: python src/train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
import yaml


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_from(value):
    return torch.device("cuda" if value == "auto" and torch.cuda.is_available() else ("cpu" if value == "auto" else value))


class ImageDomain(Dataset):
    def __init__(self, root, size):
        self.files = sorted([p for p in Path(root).glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]); self.size = size
        if not self.files: raise FileNotFoundError(f"No images found in {root}")

    def __len__(self): return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB").resize((self.size, self.size), Image.Resampling.BICUBIC)
        a = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(a).permute(2, 0, 1), self.files[i].name


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__(); self.net = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch), nn.ReLU(True), nn.ReflectionPad2d(1), nn.Conv2d(ch, ch, 3), nn.InstanceNorm2d(ch))

    def forward(self, x): return x + self.net(x)


class Generator(nn.Module):
    def __init__(self, blocks=6):
        super().__init__(); layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, 64, 7), nn.InstanceNorm2d(64), nn.ReLU(True)]
        ch = 64
        for _ in range(2): layers += [nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.InstanceNorm2d(ch * 2), nn.ReLU(True)]; ch *= 2
        layers += [ResBlock(ch) for _ in range(blocks)]
        for _ in range(2): layers += [nn.ConvTranspose2d(ch, ch // 2, 3, 2, 1, output_padding=1), nn.InstanceNorm2d(ch // 2), nn.ReLU(True)]; ch //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(ch, 3, 7), nn.Tanh()]; self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__(); layers = [nn.Conv2d(3, 64, 4, 2, 1), nn.LeakyReLU(.2, True)]; ch = 64
        for out in [128, 256, 512]: layers += [nn.Conv2d(ch, out, 4, 2, 1), nn.InstanceNorm2d(out), nn.LeakyReLU(.2, True)]; ch = out
        layers += [nn.Conv2d(ch, 1, 4, 1, 1)]; self.net = nn.Sequential(*layers)

    def forward(self, x): return self.net(x)


def save_grid(images, path, cols=4):
    images = [((x.detach().cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy() for x in images]
    if not images: return
    h, w, _ = images[0].shape; rows = math.ceil(len(images) / cols); canvas = Image.new("RGB", (w * cols, h * rows))
    for i, arr in enumerate(images): canvas.paste(Image.fromarray(arr), ((i % cols) * w, (i // cols) * h))
    canvas.save(path)


def cosine_content(a, b):
    # A deterministic low-cost content proxy: cosine similarity of average-pooled RGB vectors.
    va = a.mean((2, 3)).flatten(1); vb = b.mean((2, 3)).flatten(1); return float(nn.functional.cosine_similarity(va, vb).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="config.yaml"); args = ap.parse_args(); cfg = yaml.safe_load(Path(args.config).read_text()); seed_everything(cfg["seed"])
    root = Path.cwd(); out = root / "outputs"; ckpt = root / "checkpoints"; out.mkdir(exist_ok=True); ckpt.mkdir(exist_ok=True); (out / "pred_A2B").mkdir(exist_ok=True); (out / "pred_B2A").mkdir(exist_ok=True); (out / "recon_A").mkdir(exist_ok=True); (out / "recon_B").mkdir(exist_ok=True)
    data_root = Path(cfg["data_root"]); a_root = next((p for p in [data_root / "monet_jpg", data_root / "monet", data_root / "A"] if p.exists()), None); b_root = next((p for p in [data_root / "photo_jpg", data_root / "photo", data_root / "B"] if p.exists()), None)
    if a_root is None or b_root is None: raise FileNotFoundError("Expected Monet and photo folders under data_root")
    loader_a = DataLoader(ImageDomain(a_root, cfg["image_size"]), cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"]); loader_b = DataLoader(ImageDomain(b_root, cfg["image_size"]), cfg["batch_size"], shuffle=True, drop_last=True, num_workers=cfg["num_workers"]); iter_b = iter(loader_b)
    device = device_from(cfg["device"]); G_AB, G_BA, D_A, D_B = Generator().to(device), Generator().to(device), Discriminator().to(device), Discriminator().to(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    opt_g = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])); opt_da = torch.optim.Adam(D_A.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"])); opt_db = torch.optim.Adam(D_B.parameters(), lr=cfg["learning_rate"], betas=(cfg["beta1"], cfg["beta2"]))
    adv, cycle, ident = nn.MSELoss(), nn.L1Loss(), nn.L1Loss(); history = []; start = time.time(); grad_norms = []; nan_count = 0; images_seen = 0
    for epoch in range(1, cfg["epochs"] + 1):
        sums = {"g": 0., "d_a": 0., "d_b": 0., "cycle": 0., "identity": 0.}; batches = 0
        for real_a, _ in loader_a:
            try: real_b, names_b = next(iter_b)
            except StopIteration: iter_b = iter(loader_b); real_b, names_b = next(iter_b)
            real_a, real_b = real_a.to(device), real_b.to(device); valid_a = torch.ones_like(D_A(real_a)); valid_b = torch.ones_like(D_B(real_b)); fake_a = torch.zeros_like(valid_a); fake_b = torch.zeros_like(valid_b)
            opt_g.zero_grad(set_to_none=True); fake_b_img = G_AB(real_a); fake_a_img = G_BA(real_b); rec_a = G_BA(fake_b_img); rec_b = G_AB(fake_a_img); id_a = G_BA(real_a); id_b = G_AB(real_b)
            loss_adv = adv(D_B(fake_b_img), valid_b) + adv(D_A(fake_a_img), valid_a); loss_cycle = cycle(rec_a, real_a) + cycle(rec_b, real_b); loss_id = ident(id_a, real_a) + ident(id_b, real_b); loss_g = loss_adv + cfg["lambda_cycle"] * loss_cycle + cfg["lambda_identity"] * loss_id; nan_count += int(not torch.isfinite(loss_g).item()); loss_g.backward(); grad_norms.append(float(torch.nn.utils.clip_grad_norm_(list(G_AB.parameters()) + list(G_BA.parameters()), 5.0))); opt_g.step()
            opt_da.zero_grad(set_to_none=True); loss_da = .5 * (adv(D_A(real_a), valid_a) + adv(D_A(fake_a_img.detach()), fake_a)); loss_da.backward(); opt_da.step()
            opt_db.zero_grad(set_to_none=True); loss_db = .5 * (adv(D_B(real_b), valid_b) + adv(D_B(fake_b_img.detach()), fake_b)); loss_db.backward(); opt_db.step()
            sums["g"] += loss_g.item(); sums["d_a"] += loss_da.item(); sums["d_b"] += loss_db.item(); sums["cycle"] += loss_cycle.item(); sums["identity"] += loss_id.item(); batches += 1; images_seen += int(real_a.size(0) + real_b.size(0))
        row = {"epoch": epoch, **{k: v / max(1, batches) for k, v in sums.items()}, "elapsed_seconds": time.time() - start}; history.append(row); print(json.dumps(row), flush=True)
        if epoch == 1 or epoch % cfg["save_every"] == 0 or epoch == cfg["epochs"]:
            torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict(), "config": cfg}, ckpt / f"cyclegan_epoch_{epoch:03d}.pt")
            with torch.no_grad(): save_grid([real_a[i] for i in range(min(4, len(real_a)))] + [fake_b_img[i] for i in range(min(4, len(fake_b_img)))], out / f"samples_epoch_{epoch:03d}.png")
    G_AB.eval(); G_BA.eval(); l1_a = []; l1_b = []; content_a = []; content_b = []; audit_rows = []
    with torch.no_grad():
        for batch_i, (real_a, names_a) in enumerate(DataLoader(ImageDomain(a_root, cfg["image_size"]), 1, shuffle=False)):
            real_a = real_a.to(device); fake_b = G_AB(real_a); rec_a = G_BA(fake_b); l1_a.append(cycle(rec_a, real_a).item()); content_a.append(cosine_content(real_a, fake_b)); Image.fromarray((((fake_b[0].cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())).save(out / "pred_A2B" / names_a[0]); Image.fromarray((((rec_a[0].cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())).save(out / "recon_A" / names_a[0]); audit_rows.append({"sample_id": names_a[0], "direction": "A2B", "style_score_1": "", "style_score_2": "", "content_score_1": "", "content_score_2": "", "artifact_score_1": "", "artifact_score_2": ""})
            if batch_i >= 30: break
        for batch_i, (real_b, names_b) in enumerate(DataLoader(ImageDomain(b_root, cfg["image_size"]), 1, shuffle=False)):
            real_b = real_b.to(device); fake_a = G_BA(real_b); rec_b = G_AB(fake_a); l1_b.append(cycle(rec_b, real_b).item()); content_b.append(cosine_content(real_b, fake_a)); Image.fromarray((((fake_a[0].cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())).save(out / "pred_B2A" / names_b[0]); Image.fromarray((((rec_b[0].cpu().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).numpy())).save(out / "recon_B" / names_b[0]);
            if batch_i >= 30: break
    elapsed = time.time() - start; metrics = {"cycle_reconstruction_l1_A": float(np.mean(l1_a)), "cycle_reconstruction_l1_B": float(np.mean(l1_b)), "content_preservation_cosine_A2B": float(np.mean(content_a)), "content_preservation_cosine_B2A": float(np.mean(content_b)), "generator_gradient_norm_mean": float(np.mean(grad_norms)), "generator_gradient_norm_max": float(np.max(grad_norms)), "gradient_nan_count": nan_count, "parameter_count_generators": sum(p.numel() for p in list(G_AB.parameters()) + list(G_BA.parameters())), "parameter_count_discriminators": sum(p.numel() for p in list(D_A.parameters()) + list(D_B.parameters())), "training_time_seconds": elapsed, "images_per_second": images_seen / max(elapsed, 1e-9), "peak_memory_mb": (torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0), "device": str(device), "lpips": "run optional LPIPS evaluator", "fid_kid_precision_recall": "run approved evaluator after generation", "human_audit": "complete human_audit.csv", "kaggle_score": "fill after direct model submission"}
    (out / "history.json").write_text(json.dumps(history, indent=2)); (out / "metrics.json").write_text(json.dumps(metrics, indent=2));
    with (out / "human_audit.csv").open("w", newline="") as f: w = csv.DictWriter(f, fieldnames=audit_rows[0].keys() if audit_rows else ["sample_id"]); w.writeheader(); w.writerows(audit_rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__": main()
