"""Export a training checkpoint as the shipped recogniser: fp16 weights (13 MB instead of 27 MB), same alphabet.
fp16 storage was checked not to change accuracy (run `fp16_check` in work/results.md); inference casts back.

python export_model.py ~/dayone_local/models/crnn_v3.pt ../../models/crnn_final.pt
"""
import sys
from pathlib import Path

import torch


def main(src, dst):
    sd = torch.load(Path(src).expanduser(), map_location="cpu", weights_only=False)
    out = dict(model={k: (v.half() if v.is_floating_point() else v) for k, v in sd["model"].items()},
               chars=sd["chars"], epoch=sd.get("epoch"), val_acc=sd.get("val_acc"), tdown=sd.get("tdown", 4),
               source=Path(src).name)
    torch.save(out, Path(dst).expanduser())
    print(f"{src} -> {dst} ({Path(dst).expanduser().stat().st_size / 1e6:.1f} MB, val_acc {out['val_acc']})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
