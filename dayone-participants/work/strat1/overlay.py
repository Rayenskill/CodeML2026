"""Draw ground-truth boxes/values on a page PNG for visual spot checks (strategy 1, test T1)."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
from common import GT_DIR, LOCAL, REG  # noqa: E402


def overlay(page: int, out_dir: Path = LOCAL / "overlays") -> Path:
    g = json.loads((GT_DIR / f"page_{page:02d}.json").read_text(encoding="utf-8"))
    im = Image.open(REG / g["png"][0]).convert("RGB")
    d = ImageDraw.Draw(im)
    for f in g["fields"]:
        b = f.get("bbox")
        if not b:
            continue
        if f["type"] == "checkbox":
            d.rectangle(b, outline=(0, 160, 0) if f["value"] else (200, 120, 0), width=3)
        else:
            d.rectangle(b, outline=(255, 0, 0), width=2)
            txt = f["key"].split(".")[-2:] if f["key"].count(".") > 1 else [f["key"]]
            d.text((b[0], b[3] + 2), "/".join(txt)[:40], fill=(200, 0, 0))
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"overlay_{page:02d}.png"
    im.save(p)
    return p


if __name__ == "__main__":
    for a in sys.argv[1:]:
        print(overlay(int(a)))
