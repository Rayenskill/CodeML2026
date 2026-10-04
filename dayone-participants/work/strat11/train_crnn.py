"""Train the CRNN-CTC recogniser on generated crops.

python train_crnn.py --data ~/dayone_local/ds_v1 [--data2 ...] --out ~/dayone_local/models/crnn_v1.pt --epochs 8
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crnn import CRNN, H, Codec, to_visual  # noqa: E402

BASE_CHARS = ("0123456789/.,:;-+()'°%?! " "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
              "éèêëàâäîïôöùûüçÉÈÊÀÂÎÔÛÇœŒ–" "٠١٢٣٤٥٦٧٨٩" "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهوىي،")


def load_rows(dirs, exclude_fonts=()):
    rows = []
    for d in dirs:
        d = Path(d).expanduser()
        for line in open(d / "packed.jsonl", encoding="utf-8"):
            r = json.loads(line)
            if r.get("font") in exclude_fonts:
                continue
            r["bin"] = str(d / "crops.bin")
            rows.append(r)
    return rows


_MM = {}


def read_img(r):
    mm = _MM.get(r["bin"])
    if mm is None:
        mm = _MM[r["bin"]] = np.memmap(r["bin"], dtype=np.uint8, mode="r")
    buf = np.asarray(mm[r["off"]:r["off"] + r["len"]])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class CropDS(Dataset):
    def __init__(self, rows, codec, train=True):
        self.rows, self.codec, self.train = rows, codec, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        im = read_img(r)
        if self.train:
            im = augment(im)
        if im.shape[0] != H:
            sc = H / im.shape[0]
            im = cv2.resize(im, (max(16, int(im.shape[1] * sc)), H))
        if im.shape[1] > MAXW:
            im = cv2.resize(im, (MAXW, H), interpolation=cv2.INTER_AREA)
        text = r["text"] if r["text"] not in ("–",) else "–"
        return im, self.codec.encode(text), r["text"]


MAXW = 640


def augment(im):
    h, w = im.shape[:2]
    # random horizontal crop/pad jitter and width scale
    if random.random() < 0.5:
        sx = random.uniform(0.85, 1.15)
        im = cv2.resize(im, (max(16, int(w * sx)), h))
    if random.random() < 0.3:
        k = random.choice([3, 5])
        im = cv2.GaussianBlur(im, (k, k), 0)
    if random.random() < 0.5:
        a = random.uniform(0.75, 1.2); b = random.uniform(-25, 25)
        im = np.clip(im.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if random.random() < 0.2:
        im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
    if random.random() < 0.3:
        dy = random.randint(-4, 4)
        M = np.float32([[1, random.uniform(-0.15, 0.15), random.randint(-6, 6)], [0, 1, dy]])
        im = cv2.warpAffine(im, M, (im.shape[1], h), borderMode=cv2.BORDER_REPLICATE)
    return im


def collate(batch):
    ims, tg, txt = zip(*batch)
    w = int(math.ceil(max(i.shape[1] for i in ims) / 4) * 4)
    x = np.zeros((len(ims), H, w, 3), np.float32)
    for i, im in enumerate(ims):
        med = np.median(im.reshape(-1, 3), 0)
        x[i] = med
        x[i, :, :im.shape[1]] = im
    x = torch.from_numpy(x).permute(0, 3, 1, 2) / 127.5 - 1.0
    lens = torch.tensor([len(t) for t in tg], dtype=torch.long)
    flat = torch.tensor([c for t in tg for c in t], dtype=torch.long)
    return x, flat, lens, list(txt)


class WidthBucketSampler(Sampler):
    def __init__(self, widths, bs, shuffle=True):
        self.w, self.bs, self.shuffle = np.asarray(widths), bs, shuffle

    def __iter__(self):
        idx = np.random.permutation(len(self.w)) if self.shuffle else np.arange(len(self.w))
        chunks = [idx[i:i + self.bs * 50] for i in range(0, len(idx), self.bs * 50)]
        batches = []
        for c in chunks:
            c = c[np.argsort(self.w[c])]
            batches += [c[i:i + self.bs].tolist() for i in range(0, len(c), self.bs)]
        if self.shuffle:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return math.ceil(len(self.w) / self.bs)


def cer(a, b):
    from rapidfuzz.distance import Levenshtein
    return Levenshtein.distance(a, b) / max(1, len(b))


def evaluate(model, dl, codec, device, max_batches=None):
    model.eval()
    n = ok = 0; ce = 0.0
    with torch.no_grad():
        for bi, (x, _, _, txt) in enumerate(dl):
            lp = F.log_softmax(model(x.to(device)).float(), -1).cpu()
            for j, t in enumerate(txt):
                pred = codec.decode_greedy(lp[j])[0]
                n += 1; ok += int(pred == t); ce += cer(pred, t)
            if max_batches and bi + 1 >= max_batches:
                break
    model.train()
    return ok / max(n, 1), ce / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--init", default=None)
    ap.add_argument("--exclude_fonts", nargs="*", default=[])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tdown", type=int, default=4)
    ap.add_argument("--max_steps", type=int, default=0)
    a = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    rows = load_rows(a.data, set(a.exclude_fonts))
    chars = BASE_CHARS
    extra = sorted({c for r in rows for c in to_visual(r["text"])} - set(chars))
    chars += "".join(extra)
    if a.init:                                   # keep the initial model's alphabet (unknown chars are skipped)
        chars = torch.load(a.init, map_location="cpu", weights_only=False)["chars"]
    codec = Codec(chars)
    random.shuffle(rows)
    nval = min(3000, len(rows) // 30)
    val, tr = rows[:nval], rows[nval:]
    print(f"train {len(tr)} val {len(val)} classes {codec.n} extra={extra}", flush=True)
    widths = []
    for r in tr:
        widths.append(r.get("w") or 0)
    # widths unknown without reading images: use text length as a proxy for bucketing
    widths = [min(r["w"], MAXW) for r in tr]
    dl = DataLoader(CropDS(tr, codec, True), batch_sampler=WidthBucketSampler(widths, a.bs), num_workers=a.workers,
                    collate_fn=collate, persistent_workers=True, prefetch_factor=4)
    vdl = DataLoader(CropDS(val, codec, False), batch_size=64, num_workers=4, collate_fn=collate)
    device = "cuda"
    model = CRNN(codec.n, tdown=a.tdown).to(device)
    if a.init:
        sd = torch.load(a.init, map_location=device, weights_only=False)
        if sd["chars"] == chars:
            model.load_state_dict(sd["model"])
            print("init from", a.init)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * len(dl) if not a.max_steps else a.max_steps
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.08)
    scaler = torch.amp.GradScaler()
    step = 0; t0 = time.time()
    best = -1
    for ep in range(a.epochs):
        for x, flat, lens, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
            lp = F.log_softmax(out.float(), -1).transpose(0, 1)       # T,B,K
            T = torch.full((x.shape[0],), lp.shape[0], dtype=torch.long)
            loss = F.ctc_loss(lp, flat, T, lens, blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            step += 1
            if a.max_steps and step >= a.max_steps:
                break
            if step % 200 == 0:
                print(f"ep {ep} step {step}/{steps} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        acc, c = evaluate(model, vdl, codec, device)
        print(f"== epoch {ep}: val exact {acc:.4f} CER {c:.4f}", flush=True)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model=model.state_dict(), chars=chars, epoch=ep, val_acc=acc, tdown=a.tdown), a.out)
        if a.max_steps and step >= a.max_steps:
            break
    print("done", flush=True)


if __name__ == "__main__":
    main()
