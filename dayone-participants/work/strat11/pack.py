"""Pack a crop dataset (thousands of small JPEGs) into one binary file + offsets (Windows opens are slow).

python pack.py ~/dayone_local/ds_v1   ->  crops.bin + packed.jsonl (rows with off, len, w)
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def read(p):
    b = Path(p).read_bytes()
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_UNCHANGED)
    return b, int(im.shape[1])


def main(d):
    d = Path(d).expanduser()
    for src, dst_bin, dst_js in (("labels.jsonl", "crops.bin", "packed.jsonl"), ("checkboxes.jsonl", "cb.bin", "cb_packed.jsonl")):
        rows = [json.loads(l) for l in open(d / src, encoding="utf-8")]
        with Pool(20) as pool, open(d / dst_bin, "wb") as fb, open(d / dst_js, "w", encoding="utf-8") as fj:
            off = 0
            for r, (b, w) in zip(rows, pool.imap(read, [str(d / "img" / r["path"]) for r in rows], chunksize=64)):
                fb.write(b)
                r.update(off=off, len=len(b), w=w)
                off += len(b)
                fj.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(src, len(rows), off / 1e6, "MB", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
