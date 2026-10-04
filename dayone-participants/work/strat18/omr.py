"""Strategy 18: checkbox / mark recognition (OMR) with a small CNN on 40x40 crops (box + 10 px context).

Trained on synthetic marks (X, check marks, filled boxes; FR/AR pages; all degradations) produced by
strategy 5/11's generator. Group logic (exclusive groups) is applied afterwards in validator.py.

python omr.py train --data ~/dayone_local/ds_v1 --out ~/dayone_local/models/omr_v1.pt
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

S = 40


class OMRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.fc = nn.Linear(96, 1)

    def forward(self, x):
        return self.fc(self.f(x).flatten(1)).squeeze(1)


def prep(crop):
    im = cv2.resize(crop, (S, S), interpolation=cv2.INTER_AREA).astype(np.float32)
    # per-crop normalisation (lighting invariance)
    im = (im - im.mean()) / (im.std() + 8)
    return torch.from_numpy(im).permute(2, 0, 1)


class OMR:
    def __init__(self, path, device="cpu"):
        self.device = device
        self.net = OMRNet().to(device)
        self.net.load_state_dict(torch.load(path, map_location=device))
        self.net.eval()

    @torch.no_grad()
    def predict(self, crop) -> float:
        return float(torch.sigmoid(self.net(prep(crop)[None].to(self.device)))[0])

    @torch.no_grad()
    def predict_many(self, crops):
        x = torch.stack([prep(c) for c in crops]).to(self.device)
        return torch.sigmoid(self.net(x)).cpu().numpy()


def train(data_dirs, out, epochs=6):
    rows, ims = [], []
    for d in data_dirs:
        d = Path(d).expanduser()
        blob = (d / "cb.bin").read_bytes()
        for line in open(d / "cb_packed.jsonl", encoding="utf-8"):
            r = json.loads(line)
            buf = np.frombuffer(blob[r["off"]:r["off"] + r["len"]], np.uint8)
            ims.append(cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)); rows.append(r)
    order = list(range(len(rows)))
    random.seed(0); random.shuffle(order)
    rows = [rows[i] for i in order]
    X = torch.stack([prep(ims[i]) for i in order])
    y = torch.tensor([r["label"] for r in rows], dtype=torch.float32)
    nv = len(rows) // 10
    Xv, yv, Xt, yt = X[:nv], y[:nv], X[nv:], y[nv:]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = OMRNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-4)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 256):
            idx = perm[i:i + 256]
            xb = Xt[idx].to(dev)
            if random.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            loss = F.binary_cross_entropy_with_logits(net(xb), yt[idx].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(torch.cat([net(Xv[i:i + 1024].to(dev)) for i in range(0, len(Xv), 1024)])).cpu()
        acc = ((pv > 0.5).float() == yv).float().mean().item()
        print(f"epoch {ep} val acc {acc:.4f} (n={len(yv)}, pos rate {yv.mean():.3f})", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--data", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--epochs", type=int, default=6)
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.data, a.out, a.epochs)
