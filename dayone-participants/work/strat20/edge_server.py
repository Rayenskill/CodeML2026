"""Strategy 20: on-site edge processing box (local Wi-Fi, no internet) + central-sync stub.

Serves the phone app (strategy 15, WhatsApp-style chat) and exposes:
  POST /process               photo + idempotency key + session id -> extraction (statuses, confidences,
                              evidence crops of uncertain fields; identifiers masked before reading)
  POST /session/{sid}/finalize booklet-level consistency rules over the session's pages (strategy 7/13)
  POST /match/propose          code + non-identifying facts -> candidate patients (strategy 10)
  POST /match/decide           midwife's button -> patient id (never auto-created when a match is plausible)
  POST /records                idempotent upsert of a validated record (stands for the central server)
  GET  /records/stats          what the central store holds (demo / tests)
  GET  /records/{id}/image      the kept (identifier-masked) page image, by role; every access is logged
  GET  /health
Retries are safe: every POST is idempotent on its key. Data never leaves the facility.
Access: requests from the box itself (localhost) are trusted. Any other client pairs once (POST /pair) with a code
printed at start-up — the pairing code (role sage-femme, bound to the midwife id given at pairing) or the
supervisor code (role superviseure) — and receives a random device token (only its hash is stored) to send in
X-DayOne-Token. Failed pairing / token attempts are rate-limited per client address.

uvicorn edge_server:app --host 0.0.0.0 --port 8765   (env DAYONE_CRNN, DAYONE_OMR, DAYONE_CAL, DAYONE_STATE,
DAYONE_BOX_KEY; without DAYONE_BOX_KEY a random key is generated once into DAYONE_STATE/box.key)
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile

W = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(W / s) for s in ("shared", "strat2", "strat5", "strat6", "strat7", "strat8", "strat10", "strat11", "strat18")]

@asynccontextmanager
async def _lifespan(_app):
    print(f"[box] pairing code for the phones: {pairing_code()} | supervisor code: {supervisor_code()}", flush=True)
    yield


app = FastAPI(title="DayOne edge box", lifespan=_lifespan)
_APP_DIR = W / "strat15" / "app"         # the phone PWA (strategy 15) is served by the box on the local Wi-Fi
from common import MODELS  # noqa: E402  (repo models/ by default)
STATE = Path(os.environ.get("DAYONE_STATE", Path.home() / "dayone_local" / "edge_state"))
_lock = threading.Lock()                 # one GPU job at a time
_state_lock = threading.RLock()          # every load-modify-save of the encrypted state
CACHE_MAX = 200                          # idempotency cache (responses of the last pages), bounded
SESSIONS_MAX = 50                        # booklet sessions kept for the consistency checks
FAIL_MAX, FAIL_WINDOW = 5, 600           # failed pairing / token attempts per client address per 10 min
_cache: dict[str, dict] = {}
_sessions: dict[str, list] = {}
_proposals: dict[str, dict] = {}
_ext = None
_registry = None
PAGE_NAMES = {1: "Fiche de surveillance", 2: "Identification et antécédents", 3: "Grossesse actuelle",
              4: "Accouchement", 5: "Post-partum précoce · mère", 6: "Post-partum précoce · nouveau-né",
              7: "Post-partum tardif · mère", 8: "Post-partum tardif · nouveau-né"}


def extractor():
    global _ext
    if _ext is None:
        from calibrate import Calibrator
        from extract_zonal import Extractor
        from omr import OMR
        from recognizer import Recognizer
        rec = Recognizer(os.environ.get("DAYONE_CRNN", str(MODELS / "crnn_final.pt")))
        omr_p = os.environ.get("DAYONE_OMR", str(MODELS / "omr_v2.pt"))
        cal_p = os.environ.get("DAYONE_CAL", str(MODELS / "calibrator.json"))
        _ext = Extractor(rec, OMR(omr_p) if Path(omr_p).exists() else None,
                         Calibrator.load(cal_p) if Path(cal_p).exists() else None)
    return _ext


class Vault:
    """Encrypted-at-rest JSON documents for the box state (AES-GCM, key from DAYONE_BOX_KEY via scrypt)."""

    def __init__(self, root: Path):
        from offline import Crypto
        root.mkdir(parents=True, exist_ok=True)
        salt_f = root / "salt.bin"
        if not salt_f.exists():
            salt_f.write_bytes(os.urandom(16))
        self.root = root
        key = os.environ.get("DAYONE_BOX_KEY")
        if not key:                       # never a public default: a random key, generated once, kept beside
            kf = root / "box.key"         # the state (on a real box: move it to a USB key / TPM)
            if not kf.exists():
                kf.write_text(uuid.uuid4().hex + uuid.uuid4().hex)
                print(f"[box] DAYONE_BOX_KEY not set: generated a random state key in {kf}", flush=True)
            key = kf.read_text().strip()
        self.c = Crypto(key, salt_f.read_bytes())

    def load(self, name, default):
        f = self.root / f"{name}.bin"
        if not f.exists():
            return default
        try:
            return json.loads(self.c.dec(f.read_bytes(), name))
        except Exception as e:                      # wrong key: say so instead of an obscure crypto error
            raise RuntimeError(f"box state {f} cannot be decrypted with this key (DAYONE_BOX_KEY changed?)") from e

    def save(self, name, obj):
        tmp = self.root / f"{name}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_bytes(self.c.enc(json.dumps(obj, ensure_ascii=False).encode(), name))
        os.replace(tmp, self.root / f"{name}.bin")        # atomic: a crash never leaves a half-written file


_vault = None


def vault():
    global _vault, _cache, _sessions, _devices
    if _vault is None:
        _vault = Vault(STATE)
        for d in (_codes, _failures, _cache, _sessions):          # in-memory views of the (new) state
            d.clear()
        _devices = None
        _cache.update(_vault.load("cache", {}))
        for sid, items in _vault.load("sessions", {}).items():
            _sessions.setdefault(sid, []).extend(dict(it, lp={}) for it in items)
    return _vault


def _strip(page):
    """Page as stored in a session: no evidence crops (they are only for the live review)."""
    return dict(page, fields=[{k: v for k, v in f.items() if k != "evidence"} for f in page["fields"]])


def _persist_sessions():
    """At rest: idempotency answers without images or crops (a retried page after a box restart comes back without
    its kept image, which the phone then simply does not keep) and the last SESSIONS_MAX sessions without crops."""
    with _state_lock:
        while len(_cache) > CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        while len(_sessions) > SESSIONS_MAX:
            _sessions.pop(next(iter(_sessions)))
        vault().save("cache", {k: dict(v, page=_strip(v["page"]), masked_image=None) for k, v in _cache.items()})
        vault().save("sessions", {sid: [dict(key=it["key"], page=_strip(it["page"])) for it in items]
                                  for sid, items in _sessions.items()})


_codes: dict[str, str] = {}
_devices: dict | None = None           # sha256(device token) -> {role, midwife_id, paired_at}
_failures: dict[str, list] = {}         # client address -> times of failed attempts


def _code(name):
    """6-digit start-up codes, generated once and kept in the encrypted state (cached in memory)."""
    vault()                                 # the in-memory caches always belong to the current state
    if name not in _codes:
        with _state_lock:
            cfg = vault().load("config", {})
            if name not in cfg:
                cfg[name] = f"{int.from_bytes(os.urandom(4), 'big') % 1000000:06d}"
                vault().save("config", cfg)
            _codes[name] = cfg[name]
    return _codes[name]


def devices():
    global _devices
    vault()
    if _devices is None:
        _devices = vault().load("devices", {})
    return _devices


def _too_many_failures(host):
    import time
    now = time.time()
    _failures[host] = [t for t in _failures.get(host, []) if now - t < FAIL_WINDOW]
    return len(_failures[host]) >= FAIL_MAX


def _fail(host):
    import time
    _failures.setdefault(host, []).append(time.time())


def pairing_code():
    """6-digit code a midwife types once in the app to pair her phone with this box (role sage-femme)."""
    return _code("pairing")


def supervisor_code():
    """Code of the facility supervisor (role superviseure: kept images of every record)."""
    return _code("supervisor")


def registry():
    """Patient registry of the facility: random internal ids, registry code + non-identifying facts only."""
    global _registry
    if _registry is None:
        from privacy_linking import Registry
        _registry = Registry()
        _registry.patients = vault().load("registry", {})
    return _registry


def _save_registry():
    vault().save("registry", registry().patients)


def _crop_b64(warped, t, key, pad=(14, 10, 30, 10)):
    from crops import TEMPLATES, zone_box
    z = TEMPLATES[str(t)]["zones"][key]
    if z["type"] == "checkbox":
        x0, y0, x1, y1 = [int(v) for v in z["zone_px"]]
        box = (x0 - 60, y0 - 20, x1 + 300, y1 + 20)
    else:
        box = zone_box(t, key, pad)
    x0, y0, x1, y1 = max(0, box[0]), max(0, box[1]), box[2], box[3]
    c = warped[y0:y1, x0:x1]
    if c.size == 0:
        return None
    sc = min(1.0, 90 / c.shape[0])
    c = cv2.resize(c, (max(8, int(c.shape[1] * sc)), max(8, int(c.shape[0] * sc))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(c, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def _downscale(img, long_side=1600):
    sc = long_side / max(img.shape[:2])
    if sc >= 1:
        return img
    return cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc)), interpolation=cv2.INTER_AREA)


def masked_original(photo_bgr, page):
    """Original photo with every identifier zone (name, husband, CIN, phone, address) painted over, using the
    inverse registration homography. This is the image kept in the record; the raw photo is discarded."""
    from crops import TEMPLATES, zone_box
    from extract_zonal import is_identifier
    t = page["page_type"]
    Hinv = np.linalg.inv(np.array(page["registration"]["H"], dtype=np.float64))
    rot = page["registration"].get("rotation")
    from extract_zonal import REG_WEAK
    extra = 0 if page["registration"].get("score", 0) >= REG_WEAK else 30      # weak registration: wider masks
    out = cv2.rotate(photo_bgr, rot) if rot is not None else photo_bgr.copy()     # same turn as the registration
    for key, z in TEMPLATES[str(t)]["zones"].items():
        if z["type"] == "text" and is_identifier(key):
            x0, y0, x1, y1 = zone_box(t, key, (4 + extra, 6 + extra, 40 + extra, 6 + extra))
            pts = cv2.perspectiveTransform(np.float32([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]]), Hinv)[0]
            cv2.fillPoly(out, [pts.astype(np.int32)], (90, 90, 90))
    # printed header line "MÈRE — <name>" on postpartum pages is an identifier too
    if t in (5, 7):
        pts = cv2.perspectiveTransform(np.float32([[[195, 118], [780, 118], [780, 170], [195, 170]]]), Hinv)[0]
        cv2.fillPoly(out, [pts.astype(np.int32)], (90, 90, 90))
    ok, buf = cv2.imencode(".jpg", _downscale(out), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


API_PREFIXES = ("/process", "/session", "/match", "/records", "/dashboard", "/admin")
LOCAL_CLIENTS = ("127.0.0.1", "::1", "localhost", "testclient")


@app.middleware("http")
async def require_pairing(request, call_next):
    """Role of the caller: the box itself (localhost) or a paired device (sage-femme bound to a midwife id, or
    superviseure); any other client gets 401 on the API (static app files stay public: they hold no data)."""
    from fastapi.responses import JSONResponse
    host = request.client.host if request.client else ""
    request.state.role, request.state.midwife = None, None
    if host in LOCAL_CLIENTS:
        request.state.role = "box"
    else:
        token = request.headers.get("x-dayone-token")
        dev = devices().get(hashlib.sha256(token.encode()).hexdigest()) if token else None
        if dev:
            request.state.role, request.state.midwife = dev["role"], dev.get("midwife_id")
    if request.url.path.startswith(API_PREFIXES) and request.state.role is None:
        if _too_many_failures(host):
            return JSONResponse({"detail": "trop d'essais"}, status_code=429)
        if request.headers.get("x-dayone-token"):
            _fail(host)
        return JSONResponse({"detail": "appairage requis"}, status_code=401)
    return await call_next(request)


@app.post("/pair")
def pair(request: Request, body: dict = Body(...)):
    """Exchange a start-up code for a device token (role sage-femme bound to midwife_id, or superviseure)."""
    import secrets
    import time
    host = request.client.host if request.client else ""
    if _too_many_failures(host):
        raise HTTPException(429, "trop d'essais, réessayez plus tard")
    code = str(body.get("code", ""))
    role = "sage-femme" if code == pairing_code() else "superviseure" if code == supervisor_code() else None
    if role is None:
        _fail(host)
        raise HTTPException(403, "code incorrect")
    midwife = str(body.get("midwife_id") or "").strip()[:32] or None
    if role == "sage-femme" and not midwife:
        raise HTTPException(422, "identifiant de sage-femme requis")
    token = secrets.token_urlsafe(24)                      # 192 random bits; only its hash is kept
    with _state_lock:
        devices()[hashlib.sha256(token.encode()).hexdigest()] = dict(role=role, midwife_id=midwife, paired_at=time.time())
        vault().save("devices", devices())
    return dict(token=token, role=role, midwife_id=midwife)


@app.get("/health")
def health():
    return dict(ok=True, models=dict(crnn=os.environ.get("DAYONE_CRNN", "crnn_final.pt")))


@app.post("/process")
def process(image: UploadFile = File(...), idempotency_key: str = Form(...), midwife_id: str = Form(...),
            session_id: str = Form("default")):
    """Sync endpoint: FastAPI runs it in a worker thread, so the box keeps answering while a page is read."""
    data = image.file.read()
    vault()
    if idempotency_key in _cache:
        return _cache[idempotency_key]
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        raise HTTPException(422, "image illisible")
    from extract_zonal import RULE_KEYS, twin_type_from_content
    from privacy_linking import leak_guard
    from validator import pdate
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    with _lock:                                    # one GPU job at a time
        ext = extractor()
        page = ext.extract(rgb)
        # précoce / tardif twins: same check as the evaluation (newborn age in days; for the mother's page, the
        # gap to the delivery date read on page 4 of the same session)
        delivery = None
        for it in _sessions.get(session_id, []):
            if it["page"].get("page_type") == 4:
                delivery = pdate(next((f.get("value") for f in it["page"]["fields"]
                                       if f["key"] == "inline.date_de_l_accouchement"), None))
        tt = twin_type_from_content(page, delivery) if page.get("page_status") != "PAGE_NON_RECONNUE" else None
        if tt and tt != page["page_type"]:
            fixed = ext.extract(rgb, tt)
            fixed["twin_fix"] = f"{page['page_type']}->{tt}"
            page = fixed
        warped = ext.last_warped
        lps = {k: [x.half() for x in v] for k, v in ext.last_lp.items() if RULE_KEYS.search(k)}
    t = page["page_type"]
    recognised = page.get("page_status") != "PAGE_NON_RECONNUE"
    incidents = leak_guard(page)                   # first: a redacted value never gets an image crop
    for f in page["fields"]:
        f.pop("feats", None)
        # evidence crops only from a registered page: with a failed registration the identifier masks are not
        # where the identifiers are, so a crop could show a name
        if recognised and not f.get("redacted") and (f.get("status") in ("À_RÉVISER", "ILLISIBLE") or
                                                      (f["type"] == "checkbox" and f.get("status") != "CONNU")):
            f["evidence"] = _crop_b64(warped, t, f["key"])
    page["page_name"] = PAGE_NAMES.get(t, str(t))
    masked = masked_original(arr, page) if recognised else None
    res = dict(page=page, image_sha256=hashlib.sha256(data).hexdigest(), midwife_id=midwife_id,
               state="TRAITÉ_IA", leak_incidents=len(incidents), masked_image=masked)
    with _state_lock:
        _cache[idempotency_key] = res
        if recognised:                             # an unrecognised page never enters the booklet checks
            _sessions.setdefault(session_id, []).append(dict(key=idempotency_key, page=page, lp=lps))
        _persist_sessions()
    return res


@app.post("/session/{sid}/finalize")
def finalize(sid: str, body: dict = Body(default={})):
    """Booklet-level rules across the pages of one session (dates, ages, same fact on two pages), applied to
    the pages as reviewed by the midwife (body.pages: {idempotency_key: page}); values she confirmed are
    never changed silently, only flagged with a suggestion."""
    from validator import apply_form_logic, validate_booklet
    lps = {it["key"]: it["lp"] for it in _sessions.get(sid, [])}
    pages = body.get("pages") or {it["key"]: it["page"] for it in _sessions.get(sid, [])}
    for k, p in pages.items():
        p["_lp"] = lps.get(k, {})
        apply_form_logic(p)                        # again, on the boxes as corrected by the midwife
    with _lock:
        issues = validate_booklet(list(pages.values()), extractor().rec)
    for p in pages.values():
        p.pop("_lp", None)
    return dict(issues=issues, pages=pages)


@app.post("/match/propose")
def match_propose(body: dict = Body(...)):
    reg = registry()
    prop = reg.propose(body.get("code", ""), body.get("facts", {}))
    pid = uuid.uuid4().hex
    with _state_lock:
        _proposals.update(vault().load("proposals", {}))
        _proposals[pid] = dict(prop, code=body.get("code", ""), facts=body.get("facts", {}))
        vault().save("proposals", _proposals)             # a box restart never strands a pending decision
    view = [dict(rank=i + 1, code=c["code"], dist=round(c["dist"], 2), agree=c["agree"], clash=c["clash"],
                 facts={k: reg.patients[c["pid"]]["facts"].get(k) for k in ("ddr", "date_prevue")})
            for i, c in enumerate(prop["candidates"])]
    return dict(proposal_id=pid, kind=prop["kind"], candidates=view, buttons=prop["buttons"])


@app.post("/match/decide")
def match_decide(body: dict = Body(...)):
    prop = _proposals.get(body["proposal_id"]) or vault().load("proposals", {}).get(body["proposal_id"])
    if prop is None:
        raise HTTPException(404, "proposition inconnue")
    with _state_lock:
        pid, state = registry().decide(prop, body["choice"], prop["code"], prop["facts"])
        _save_registry()
    return dict(patient_id=pid, state=state)


def _img_doc(record_id: str, version: int) -> str:
    """Vault document holding one kept image (file name derived from the id, whatever characters it has)."""
    return f"img_{hashlib.sha256(record_id.encode()).hexdigest()[:24]}_{version}"


@app.post("/records")
def upsert_record(body: dict = Body(...)):
    """Central store stub: idempotent on record_id:version; refuses identifier values (defence in depth)."""
    from privacy_linking import leak_guard
    key = f"{body['record_id']}:{body.get('version', 0)}"
    with _state_lock:
        central = vault().load("central", [])
        if any(r["key"] == key for r in central):
            return dict(ok=True, duplicate=True)
        leaks = sum(len(leak_guard(p)) for p in body.get("pages", []))
        image = body.get("image")                  # kept page image (identifiers already masked by the box)
        if image:
            vault().save(_img_doc(body["record_id"], int(body.get("version", 0))), dict(data=image))
        central.append(dict(key=key, record_id=body["record_id"], version=int(body.get("version", 0)),
                            patient_id=body.get("patient_id"), pages=body.get("pages", []), leaks_removed=leaks,
                            midwife_id=body.get("midwife_id"), captured_at=body.get("captured_at"),
                            image_sha256=body.get("image_sha256"), has_image=bool(image)))
        vault().save("central", central)
    return dict(ok=True, duplicate=False, leaks_removed=leaks)


@app.get("/records/{record_id}/image")
def record_image(record_id: str, request: Request):
    """The original page image kept with the record (identifier zones masked), with role-based access: the
    supervisor and the box see every record, a midwife only the records she captured. Every access is logged."""
    import time
    role = getattr(request.state, "role", None)
    who = getattr(request.state, "midwife", None) or ""         # bound to the device token at pairing
    with _state_lock:
        rows = [r for r in vault().load("central", []) if r["record_id"] == record_id and r.get("has_image")]
        if not rows:
            raise HTTPException(404, "aucune image pour ce dossier")
        r = max(rows, key=lambda x: x.get("version", 0))
        allowed = role in ("box", "superviseure") or (role == "sage-femme" and who and who == r.get("midwife_id"))
        log = vault().load("access_log", [])
        log.append(dict(record_id=record_id, role=role, midwife=who or None, allowed=bool(allowed), at=time.time()))
        vault().save("access_log", log)
        if not allowed:
            raise HTTPException(403, "accès réservé à la sage-femme du dossier ou à la superviseure")
        img = vault().load(_img_doc(record_id, r["version"]), {}).get("data")
    return dict(record_id=record_id, version=r["version"], image=img, captured_at=r.get("captured_at"),
                midwife_id=r.get("midwife_id"))


@app.get("/records/stats")
def record_stats():
    rows = vault().load("central", [])
    return dict(records=len(rows), unique_records=len({r["record_id"] for r in rows}),
                patients=len(registry().patients))


@app.get("/records/previous")
def previous_record(patient_id: str, page_type: int, exclude: str = ""):
    """Strategy 13: last validated version of this page for this patient (re-digitisation diff)."""
    rows = [r for r in vault().load("central", []) if r["patient_id"] == patient_id and r["record_id"] != exclude
            and any(p.get("page_type") == page_type for p in r["pages"])]
    if not rows:
        return dict(found=False)
    page = next(p for p in rows[-1]["pages"] if p.get("page_type") == page_type)
    return dict(found=True, record_id=rows[-1]["record_id"], page=page)


@app.get("/dashboard")
def dashboard():
    """Strategy 19: anonymised aggregates from the synced records (k=5 suppression, missing counts shown)."""
    from fastapi.responses import HTMLResponse
    sys.path.insert(0, str(W / "strat19"))
    from dashboard import by_age, html, indicators, indicators_from_records, load
    rows = vault().load("central", [])
    rec = indicators_from_records(rows)
    page = html(indicators(load()), by_age(load()))
    page = page.replace("<h1>", f"<h2>Dossiers numérisés ({len(rows)} pages synchronisées)</h2>{rec.to_html(index=False)}<h1>", 1)
    return HTMLResponse(page)


@app.post("/admin/seed")
def seed(request: Request, body: dict = Body(...)):
    """Demo helper: pre-register patients (code + facts) as if they had been seen at earlier visits. Idempotent
    per code, so rehearsing the demo twice does not create look-alike patients. Only from the box itself."""
    from privacy_linking import norm_code
    if getattr(request.state, "role", None) != "box":
        raise HTTPException(403, "réservé à la box")
    with _state_lock:
        reg = registry()
        known = {p["code"] for p in reg.patients.values()}
        for p in body.get("patients", []):
            if norm_code(p["code"]) not in known:
                reg.create(p["code"], p.get("facts", {}))
                known.add(norm_code(p["code"]))
        _save_registry()
    return dict(patients=len(reg.patients))


if _APP_DIR.exists():
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_APP_DIR), html=True), name="pwa")
