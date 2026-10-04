"""Local OCR for pages whose text is drawn as vector strokes or raster images.

Many shop drawings are plotted from CAD with stroke fonts: the PDF holds no
text at all.  Those pages are rendered and read with RapidOCR (PP-OCR models
running on onnxruntime, bundled in the wheel).  Everything runs on the local
machine; nothing leaves it.

Sheets are large (36 x 24 in) and their lettering small, so the page is rendered
at `ocr_dpi` and read tile by tile, with an overlap wide enough to hold a whole
callout.  Results are cached on disk per (file, page, dpi).
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pymupdf

from ..config import Config
from ..models import TextLine
from .vector import lettering_image


def ocr_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def gpu_available(cfg: Config) -> bool:
    """True when OCR can run on the local GPU (Windows DirectML build of onnxruntime)."""
    if cfg.ocr_gpu == "off":
        return False
    try:
        import onnxruntime

        return "DmlExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


def _legacy_backend(cfg: Config, threads: int):
    """PP-OCRv4 mobile models bundled in the rapidocr-onnxruntime wheel (no download)."""
    from rapidocr_onnxruntime import RapidOCR

    kwargs = {"max_side_len": max(cfg.ocr_tile, 2000), "text_score": cfg.ocr_min_score}
    if threads:
        kwargs["intra_op_num_threads"] = threads
    if gpu_available(cfg):
        kwargs.update(det_use_dml=True, cls_use_dml=True, rec_use_dml=True)
    engine = RapidOCR(**kwargs)

    def run(img: np.ndarray):
        result, _ = engine(img)
        return [(box, text, score) for box, text, score in result or []]

    return run


def _modern_backend(cfg: Config, threads: int):
    """PP-OCRv5 / v6 models through rapidocr 3.x (fetched once on first use, then cached locally)."""
    from rapidocr import RapidOCR
    from rapidocr.utils.typings import LangRec, ModelType, OCRVersion

    version = OCRVersion[f"PPOCR{cfg.ocr_model.upper()}"]
    params = {
        "Global.text_score": cfg.ocr_min_score,
        "Global.max_side_len": max(cfg.ocr_tile, 2000),
        "Global.log_level": "error",
        "Det.ocr_version": OCRVersion[f"PPOCR{(cfg.ocr_det_model or cfg.ocr_model).upper()}"],
        "Det.model_type": ModelType[cfg.ocr_det_type.upper()],
        "Rec.ocr_version": version,
        "Rec.model_type": ModelType[cfg.ocr_rec_type.upper()],
        "Rec.lang_type": LangRec[cfg.ocr_rec_lang.upper()],
        "EngineConfig.onnxruntime.use_dml": gpu_available(cfg),
    }
    if threads:
        params["EngineConfig.onnxruntime.intra_op_num_threads"] = threads
    if cfg.ocr_model_dir:
        params["Global.model_root_dir"] = cfg.ocr_model_dir
    engine = RapidOCR(params=params)

    def run(img: np.ndarray):
        result = engine(img)
        if result is None or result.boxes is None:
            return []
        return list(zip(result.boxes, result.txts, result.scores))

    return run


class OcrEngine:
    def __init__(self, cfg: Config, threads: int = 0):
        self.cfg = cfg
        if cfg.ocr_model == "v4":
            self._run = _legacy_backend(cfg, threads)
        else:
            try:
                self._run = _modern_backend(cfg, threads)
            except Exception:
                # rapidocr 3.x missing, or its models not on disk and no network: the bundled v4 models always work.
                self._run = _legacy_backend(cfg, threads)

    def read_page(self, page: pymupdf.Page) -> list[TextLine]:
        scale = self.cfg.ocr_dpi / 72.0
        img = lettering_image(page, scale, self.cfg.ocr_glyph_max_pt) if self.cfg.ocr_lettering_only else None
        if img is None:  # a scan, or a page without character-sized paths: read the plain raster
            pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
        if self.cfg.ocr_thicken > 0:
            # CAD stroke fonts are hairlines; a grey-level erosion thickens dark strokes.
            import cv2

            size = 2 * self.cfg.ocr_thicken + 1
            img = cv2.erode(img, np.ones((size, size), np.uint8))
        return self.read_image(img, scale)

    def read_image(self, img: np.ndarray, scale: float) -> list[TextLine]:
        """OCR a grayscale image; boxes are returned in points (pixels / scale)."""
        tile, overlap = self.cfg.ocr_tile, self.cfg.ocr_overlap
        stride = max(tile - overlap, 1)
        h, w = img.shape[:2]
        raw: list[tuple[tuple[float, float, float, float], str, float]] = []
        for ty in _starts(h, tile, stride):
            for tx in _starts(w, tile, stride):
                sub = img[ty: ty + tile, tx: tx + tile]
                if (sub < 160).mean() < 0.0004:  # blank tile
                    continue
                for box, text, score in self._run(np.ascontiguousarray(np.stack([sub] * 3, axis=-1))):
                    xs = [float(p[0]) + tx for p in box]  # plain floats: the backends return numpy scalars
                    ys = [float(p[1]) + ty for p in box]
                    raw.append(((min(xs), min(ys), max(xs), max(ys)), str(text).strip(), float(score)))
        lines = []
        for (x0, y0, x1, y1), text, score in _dedupe(raw):
            if not text:
                continue
            bw, bh = x1 - x0, y1 - y0
            vertical = bh > 1.5 * bw and len(text) > 1
            lines.append(
                TextLine(
                    text=text,
                    bbox=(x0 / scale, y0 / scale, x1 / scale, y1 / scale),
                    size=(bw if vertical else bh) / scale,
                    dx=0.0 if vertical else 1.0,
                    dy=-1.0 if vertical else 0.0,  # CAD convention: vertical text reads bottom to top
                    origin="ocr",
                    conf=score,
                )
            )
        return lines


def join_row_fragments(lines: list[TextLine], gap: float = 1.5) -> list[TextLine]:
    """Glue back the pieces of one line of lettering that the OCR detector boxed separately.

    The detector cuts a line at wide spaces: "24 10M 10A12" and, further right, "23@300"
    (23 spaces at 300 mm) come out as two boxes, and the spacing is lost to the callout.
    Horizontal pieces of the same height, on the same row, closer than `gap` times their
    height, are read as one line.
    """
    flat = sorted((l for l in lines if not l.vertical), key=lambda l: l.bbox[0])
    rows: list[list[TextLine]] = []
    by_band: dict[int, list[int]] = {}  # rows indexed by the height band of their first piece
    for line in flat:
        h = line.bbox[3] - line.bbox[1]
        cy = (line.bbox[1] + line.bbox[3]) / 2
        band = int(cy // 4.0)
        for idx in (i for b in range(band - 3, band + 4) for i in by_band.get(b, ())):
            last = rows[idx][-1]
            lh = last.bbox[3] - last.bbox[1]
            if (abs(cy - (last.bbox[1] + last.bbox[3]) / 2) <= 0.35 * min(h, lh) and 0.65 <= h / max(lh, 1e-6) <= 1.5
                    and -0.3 * min(h, lh) <= line.bbox[0] - last.bbox[2] <= gap * max(h, lh)):
                rows[idx].append(line)
                break
        else:
            by_band.setdefault(band, []).append(len(rows))
            rows.append([line])
    out = [l for l in lines if l.vertical]
    for row in rows:
        if len(row) == 1:
            out.append(row[0])
            continue
        out.append(TextLine(
            text=" ".join(l.text for l in row),
            bbox=(row[0].bbox[0], min(l.bbox[1] for l in row), row[-1].bbox[2], max(l.bbox[3] for l in row)),
            size=max(l.size for l in row), dx=1.0, dy=0.0, origin="ocr", conf=min(l.conf for l in row),
        ))
    return out


def _starts(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _dedupe(items: list[tuple[tuple[float, float, float, float], str, float]]):
    """Drop duplicates from overlapping tiles: of two boxes covering the same
    ink, keep the one with the longer text (the uncut one), then the better score."""
    items = sorted(items, key=lambda it: (-len(it[1]), -it[2]))
    kept: list[tuple[tuple[float, float, float, float], str, float]] = []
    grid: dict[tuple[int, int], list[int]] = {}
    cell = 256.0
    for box, text, score in items:
        x0, y0, x1, y1 = box
        area = max((x1 - x0) * (y1 - y0), 1e-6)
        cells = [(cx, cy) for cx in range(int(x0 // cell), int(x1 // cell) + 1)
                 for cy in range(int(y0 // cell), int(y1 // cell) + 1)]
        duplicate = False
        for c in cells:
            for idx in grid.get(c, ()):
                kx0, ky0, kx1, ky1 = kept[idx][0]
                iw = min(x1, kx1) - max(x0, kx0)
                ih = min(y1, ky1) - max(y0, ky0)
                if iw > 0 and ih > 0 and iw * ih / area > 0.6:
                    duplicate = True
                    break
            if duplicate:
                break
        if duplicate:
            continue
        kept.append((box, text, score))
        for c in cells:
            grid.setdefault(c, []).append(len(kept) - 1)
    return kept


# ------------------------------------------------------------------ cache

def cache_path(cache_dir: Path, pdf_path: Path, page_index: int, cfg: Config) -> Path:
    st = pdf_path.stat()
    key = hashlib.sha1(f"{pdf_path.name}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()[:16]
    model = f"{cfg.ocr_det_model or cfg.ocr_model}{cfg.ocr_det_type}-{cfg.ocr_model}{cfg.ocr_rec_type}{cfg.ocr_rec_lang}"
    return cache_dir / f"{key}_p{page_index + 1}_{cfg.ocr_dpi}dpi_t{cfg.ocr_thicken}_{model}{'_L' if cfg.ocr_lettering_only else ''}.json"


def load_cached(path: Path) -> Optional[list[TextLine]]:
    if not path.exists():
        return None
    try:
        return [TextLine(text=d["t"], bbox=tuple(d["b"]), size=d["s"], dx=d["dx"], dy=d["dy"], origin="ocr",
                         conf=d["c"]) for d in json.loads(path.read_text(encoding="utf-8"))]
    except (ValueError, KeyError):
        return None


def save_cached(path: Path, lines: list[TextLine]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"t": l.text, "b": [round(v, 2) for v in l.bbox], "s": round(l.size, 2), "dx": l.dx, "dy": l.dy,
             "c": round(l.conf, 3)} for l in lines]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


# ------------------------------------------------------------- batch (parallel)

_WORKER_ENGINE: Optional[OcrEngine] = None


def _init_worker(cfg: Config, threads: int) -> None:
    global _WORKER_ENGINE
    _WORKER_ENGINE = OcrEngine(cfg, threads=threads)


def _ocr_task(task: tuple[str, int, str]) -> int:
    """Lines read on one page, or -1 when that page could not be read (it is then left out, not fatal)."""
    pdf_path, page_index, out_path = task
    assert _WORKER_ENGINE is not None
    try:
        with pymupdf.open(pdf_path) as doc:
            lines = _WORKER_ENGINE.read_page(doc[page_index])
    except Exception:  # one unreadable page must not stop the analysis of the project
        return -1
    save_cached(Path(out_path), lines)
    return len(lines)


def ocr_pages(tasks: Iterable[tuple[Path, int]], cfg: Config, cache_dir: Path, progress=None) -> int:
    """OCR every (pdf, page) not yet in the cache, in parallel.  Returns pages read."""
    todo = []
    for pdf_path, page_index in tasks:
        out = cache_path(cache_dir, pdf_path, page_index, cfg)
        if not out.exists():
            todo.append((str(pdf_path), page_index, str(out)))
    if not todo:
        return 0
    # One GPU is shared best by a couple of processes; on CPU, spread over the cores.
    workers = max(1, min(cfg.workers, 2 if gpu_available(cfg) else cfg.workers, len(todo)))
    threads = max(1, (os.cpu_count() or 2) // workers)
    done = 0
    if workers > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(cfg, threads)) as pool:
                for _ in pool.map(_ocr_task, todo):
                    done += 1
                    if progress:
                        progress(done, len(todo))
            return done
        except BrokenProcessPool:
            # A worker died (out of memory, GPU driver reset).  Finish one page at a time, each in a
            # worker of its own, so that a page that kills its worker is skipped, not the whole run.
            todo = [task for task in todo if not Path(task[2]).exists()]
            total = done + len(todo)
            while todo:
                try:
                    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker,
                                             initargs=(cfg, max(1, os.cpu_count() or 1))) as pool:
                        while todo:
                            pool.submit(_ocr_task, todo[0]).result()
                            todo.pop(0)
                            done += 1
                            if progress:
                                progress(done, total)
                except BrokenProcessPool:
                    todo.pop(0)  # this page brought its worker down: left out, retried on the next run
                    done += 1
            return done
    _init_worker(cfg, threads)
    for task in todo:
        _ocr_task(task)
        done += 1
        if progress:
            progress(done, len(todo))
    return done
