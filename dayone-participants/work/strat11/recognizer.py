"""Inference wrapper for the CRNN: batched recognition + CTC rescoring of candidate strings."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crnn import CRNN, H, Codec, preprocess  # noqa: E402

MAXW = 640


class Recognizer:
    def __init__(self, path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.codec = Codec(sd["chars"])
        self.tdown = sd.get("tdown", 4)
        self.model = CRNN(self.codec.n, tdown=self.tdown).to(self.device)
        self.model.load_state_dict(sd["model"])
        self.model.eval()

    @torch.no_grad()
    def logprobs(self, crops: list[np.ndarray], bs=64):
        """crops: RGB arrays -> list of (T x K) CPU log-prob tensors."""
        ims = []
        for c in crops:
            im = preprocess(c, H)
            if im.shape[1] > MAXW:
                import cv2
                im = cv2.resize(im, (MAXW, H), interpolation=cv2.INTER_AREA)
            ims.append(im)
        order = np.argsort([i.shape[1] for i in ims])
        out = [None] * len(ims)
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            batch = [ims[i] for i in idx]
            w = int(np.ceil(max(b.shape[1] for b in batch) / 4) * 4)
            x = np.zeros((len(batch), H, w, 3), np.float32)
            for j, b in enumerate(batch):
                x[j] = np.median(b.reshape(-1, 3), 0)
                x[j, :, :b.shape[1]] = b
            xt = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device) / 127.5 - 1
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                lo = self.model(xt)
            lp = F.log_softmax(lo.float(), -1).cpu()
            for j, i in enumerate(idx):
                tlen = int(np.ceil(batch[j].shape[1] / self.tdown))
                out[i] = lp[j, :max(tlen, 1)]
        return out

    def logprobs_tta(self, crops: list[np.ndarray], bs=64):
        """Test-time augmentation: average CTC posteriors over photometric / small vertical-shift variants
        (same width, so the time axes align)."""
        import cv2
        variants = []
        for c in crops:
            v = [c]
            v.append(np.clip(c.astype(np.float32) * 1.25 - 30, 0, 255).astype(np.uint8))       # more contrast
            M = np.float32([[1, 0, 0], [0, 1, -3]])
            v.append(cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), borderMode=cv2.BORDER_REPLICATE))
            M = np.float32([[1, 0, 0], [0, 1, 3]])
            v.append(cv2.warpAffine(c, M, (c.shape[1], c.shape[0]), borderMode=cv2.BORDER_REPLICATE))
            variants.append(v)
        k = len(variants[0])
        flat = [x for v in variants for x in v]
        lps = self.logprobs(flat, bs)
        out = []
        for i in range(len(crops)):
            group = lps[i * k:(i + 1) * k]
            T = min(g.shape[0] for g in group)
            p = torch.stack([g[:T].exp() for g in group]).mean(0)
            out.append(p.clamp_min(1e-12).log())
        return out

    def decode(self, lp):
        return self.codec.decode_greedy(lp)

    def score(self, lp, text: str) -> float:
        if isinstance(lp, (list, tuple)):          # several views of the same crop: mean log-likelihood
            return sum(self.score(x, text) for x in lp) / len(lp)
        from crnn import ctc_string_logprob
        lp = lp.float()
        tg = self.codec.encode(text)
        if len(tg) * 1 > lp.shape[0]:
            return -1e9
        return ctc_string_logprob(lp, tg)


class EnsembleRecognizer(Recognizer):
    """Posterior averaging over several CRNN checkpoints sharing one alphabet (strategy 12, same-engine ensemble)."""

    def __init__(self, paths, device=None):
        self.members = [Recognizer(p, device) for p in paths]
        assert all(m.codec.chars == self.members[0].codec.chars for m in self.members)
        self.codec = self.members[0].codec
        self.device = self.members[0].device

    def logprobs(self, crops, bs=64):
        outs = [m.logprobs(crops, bs) for m in self.members]
        res = []
        for group in zip(*outs):
            T = min(g.shape[0] for g in group)
            res.append(torch.stack([g[:T].exp() for g in group]).mean(0).clamp_min(1e-12).log())
        return res
