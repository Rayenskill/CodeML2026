"""Strategy 11: small CRNN-CTC handwriting/print recogniser for registry cells (FR / EN / AR).

Labels are stored in logical order; the network reads left-to-right, so Arabic labels are converted
to visual order with the Unicode bidi algorithm for training, and predictions converted back.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from bidi.algorithm import get_display
except ImportError:  # python-bidi >= 0.5
    from bidi import get_display

H = 64


def has_rtl(s: str) -> bool:
    return any("֐" <= c <= "ࣿ" for c in s)


def to_visual(s: str) -> str:
    return get_display(s) if has_rtl(s) else s


def to_logical(s: str) -> str:
    return get_display(s) if has_rtl(s) else s


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.sc = None if (cin == cout and stride == 1) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False),
                                                                          nn.BatchNorm2d(cout))

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.sc is None else self.sc(x)))


class CRNN(nn.Module):
    def __init__(self, n_classes: int, c=(32, 96, 192, 256), rnn=192, tdown: int = 4):
        """tdown = horizontal downsampling (4: T=W/4; 2: T=W/2, twice the CTC frames for thin doubled glyphs).
        Parameter shapes do not depend on tdown, so a model can be fine-tuned from the other variant."""
        super().__init__()
        c = list(c)
        self.tdown = tdown
        self.stem = nn.Sequential(nn.Conv2d(3, c[0], 3, 1, 1, bias=False), nn.BatchNorm2d(c[0]), nn.ReLU(),
                                  nn.MaxPool2d(2))                                   # 32 x W/2
        self.l1 = nn.Sequential(Block(c[0], c[1]), nn.MaxPool2d(2 if tdown == 4 else (2, 1)))  # 16 x W/tdown
        self.l2 = nn.Sequential(Block(c[1], c[2]), Block(c[2], c[2]), nn.MaxPool2d((2, 1)))  # 8 x W/4
        self.l3 = nn.Sequential(Block(c[2], c[3]), Block(c[3], c[3]), nn.MaxPool2d((2, 1)))  # 4 x W/4
        self.l4 = nn.Sequential(Block(c[3], c[3]))
        self.proj = nn.Conv2d(c[3], 256, (4, 1))                                          # 1 x W/4
        self.rnn = nn.LSTM(256, rnn, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(2 * rnn, n_classes)

    def forward(self, x):
        x = self.l4(self.l3(self.l2(self.l1(self.stem(x)))))
        x = F.relu(self.proj(x)).squeeze(2).transpose(1, 2)    # B, T, C
        x, _ = self.rnn(x)
        return self.fc(x)                                       # B, T, K (K-1 = blank? no: 0 = blank)


class Codec:
    def __init__(self, chars: str):
        self.chars = chars
        self.idx = {c: i + 1 for i, c in enumerate(chars)}     # 0 = CTC blank

    @property
    def n(self):
        return len(self.chars) + 1

    def encode(self, s: str):
        return [self.idx[c] for c in to_visual(s) if c in self.idx]

    def decode_greedy(self, logp: torch.Tensor):
        """logp: T x K log-probs -> (text_logical, char_probs list, seq_conf)"""
        p = logp.exp()
        best = p.argmax(-1).tolist()
        bp = p.max(-1).values.tolist()
        out, probs, prev = [], [], 0
        for k, pr in zip(best, bp):
            if k != prev and k != 0:
                out.append(self.chars[k - 1]); probs.append(pr)
            elif k == prev and k != 0 and probs:
                probs[-1] = max(probs[-1], pr)
            prev = k
        # sequence confidence: product of the per-timestep max probabilities along the path
        conf = float(np.exp(logp.max(-1).values.sum().item()))
        return to_logical("".join(out)), probs, conf


def ctc_string_logprob(logp: torch.Tensor, target: list[int]) -> float:
    """log P(target | x) under CTC (forward algorithm) for rescoring candidate values."""
    if not target:
        return float(logp[:, 0].sum())
    lp = logp.unsqueeze(1)                                     # T,1,K
    t = torch.tensor([target], dtype=torch.long)
    loss = F.ctc_loss(lp, t, torch.tensor([lp.shape[0]]), torch.tensor([len(target)]), blank=0,
                      reduction="sum", zero_infinity=False)
    return -float(loss)


def preprocess(rgb: np.ndarray, h: int = H) -> np.ndarray:
    import cv2
    sc = h / rgb.shape[0]
    w = max(16, int(round(rgb.shape[1] * sc)))
    im = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    return im


def batchify(imgs: list[np.ndarray], device):
    w = max(i.shape[1] for i in imgs)
    w = int(math.ceil(w / 4) * 4)
    x = np.full((len(imgs), H, w, 3), 255, np.float32)
    for i, im in enumerate(imgs):
        x[i, :, :im.shape[1]] = im
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(device)
    return (x / 127.5) - 1.0
