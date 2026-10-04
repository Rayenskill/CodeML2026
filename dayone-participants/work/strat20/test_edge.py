"""Edge box API test with a stub extractor (the real model is exercised in run_eval)."""
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))
import edge_server  # noqa: E402


class Stub:
    calls = 0
    last_lp = {}

    def extract(self, rgb):
        import numpy as np
        Stub.calls += 1
        self.last_warped = np.full((2339, 1654, 3), 200, np.uint8)
        return dict(page_type=3, page_status="PAGE_NON_RECONNUE", registration=dict(score=0.1, H=np.eye(3).tolist()),
                    fields=[dict(key="inline.ddr", type="text", value="26/04/2025", status="CONNU", feats={}),
                            dict(key="inline.cin", type="text", value="CB609814", status="CONNU")])


def test_process_idempotent_and_leak_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    edge_server._vault = None
    edge_server._ext = Stub()
    c = TestClient(edge_server.app)
    img = cv2.imencode(".jpg", np.full((100, 70, 3), 200, np.uint8))[1].tobytes()
    r1 = c.post("/process", files={"image": ("p.jpg", img, "image/jpeg")}, data={"idempotency_key": "rec1:1", "midwife_id": "sf-01"})
    assert r1.status_code == 200
    body = r1.json()
    assert body["state"] == "TRAITÉ_IA" and body["leak_incidents"] == 1
    assert [f for f in body["page"]["fields"] if f["key"] == "inline.cin"][0]["value"] is None
    # unrecognised page: no evidence crop (masks unreliable), no kept image, kept out of the booklet checks
    assert all("evidence" not in f for f in body["page"]["fields"]) and body["masked_image"] is None
    assert not any(it["key"] == "rec1:1" for it in edge_server._sessions.get("default", []))
    r2 = c.post("/process", files={"image": ("p.jpg", img, "image/jpeg")}, data={"idempotency_key": "rec1:1", "midwife_id": "sf-01"})
    assert r2.json() == body and Stub.calls == 1          # retry served from cache, processed once
    bad = c.post("/process", files={"image": ("p.jpg", b"nope", "image/jpeg")}, data={"idempotency_key": "x", "midwife_id": "sf"})
    assert bad.status_code == 422


def test_match_and_records_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    edge_server._vault = None; edge_server._registry = None
    c = TestClient(edge_server.app)
    c.post("/admin/seed", json={"patients": [{"code": "2026-823-001", "facts": {"ddr": "26/04/2025"}}]})
    p = c.post("/match/propose", json={"code": "2026-823-007", "facts": {"ddr": "26/04/2025"}}).json()
    assert p["candidates"] and "Aucune, créer" in p["buttons"] and "Je ne sais pas" in p["buttons"]
    d = c.post("/match/decide", json={"proposal_id": p["proposal_id"], "choice": "Patiente 1"}).json()
    assert d["state"] == "PATIENTE_LIÉE" and d["patient_id"]
    body = {"record_id": "r1", "version": 1, "patient_id": d["patient_id"],
            "pages": [{"page_type": 3, "fields": [{"key": "inline.ddr", "value": "26/04/2025"}, {"key": "inline.cin", "value": "CB609814"}]}]}
    a, b = c.post("/records", json=body).json(), c.post("/records", json=body).json()
    assert a["duplicate"] is False and a["leaks_removed"] == 1 and b["duplicate"] is True
    assert c.get("/records/stats").json()["unique_records"] == 1
    prev = c.get(f"/records/previous?patient_id={d['patient_id']}&page_type=3&exclude=other").json()
    assert prev["found"] and prev["page"]["fields"][1]["value"] is None          # identifier never stored
    raw = b"".join(f.read_bytes() for f in tmp_path.glob("*.bin"))
    assert b"26/04/2025" not in raw and b"2026-823" not in raw                    # box state encrypted at rest


class StubOK(Stub):
    def extract(self, rgb, page_type=None):
        self.last_warped = np.full((2339, 1654, 3), 200, np.uint8)
        return dict(page_type=3, page_status="OK", registration=dict(score=0.9, H=np.eye(3).tolist()),
                    fields=[dict(key="inline.ddr", type="text", value="26/04/2025", status="À_RÉVISER", feats={}),
                            dict(key="inline.taille", type="text", value="155 cm", status="CONNU", feats={})])


def test_recognised_page_evidence_and_masked_image(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    edge_server._vault = None
    edge_server._ext = StubOK()
    c = TestClient(edge_server.app)
    img = cv2.imencode(".jpg", np.full((2339, 1654, 3), 230, np.uint8))[1].tobytes()
    body = c.post("/process", files={"image": ("p.jpg", img, "image/jpeg")},
                  data={"idempotency_key": "ok:1", "midwife_id": "sf-01", "session_id": "s1"}).json()
    f = {x["key"]: x for x in body["page"]["fields"]}
    assert f["inline.ddr"]["evidence"].startswith("data:image/jpeg") and "evidence" not in f["inline.taille"]
    assert body["masked_image"].startswith("data:image/jpeg")
    sess = edge_server._vault.load("sessions", {})["s1"]            # stored for the booklet checks, without crops
    assert sess[0]["key"] == "ok:1" and all("evidence" not in x for x in sess[0]["page"]["fields"])


def _pair(client, code, midwife=None):
    return client.post("/pair", json={"code": code, "midwife_id": midwife})


def test_pairing_required_from_the_wifi(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    edge_server._vault = None; edge_server._registry = None
    phone = TestClient(edge_server.app, client=("192.168.1.20", 50000))
    assert phone.get("/health").status_code == 200
    assert phone.get("/records/stats").status_code == 401
    r = _pair(phone, edge_server.pairing_code(), "sf-01")
    assert r.status_code == 200 and r.json()["role"] == "sage-femme"
    token = r.json()["token"]
    assert token != edge_server.pairing_code() and len(token) >= 32           # a random device token, not the code
    assert phone.get("/records/stats", headers={"X-DayOne-Token": token}).status_code == 200
    assert phone.get("/records/stats", headers={"X-DayOne-Token": edge_server.pairing_code()}).status_code == 401
    assert phone.post("/admin/seed", headers={"X-DayOne-Token": token}, json={"patients": []}).status_code == 403
    assert TestClient(edge_server.app).get("/records/stats").status_code == 200        # the box itself
    raw = b"".join(f.read_bytes() for f in tmp_path.glob("*.bin"))
    assert token.encode() not in raw                                                   # only a hash is stored


def test_pairing_attempts_are_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    edge_server._vault = None
    intruder = TestClient(edge_server.app, client=("192.168.1.66", 50000))
    good = edge_server.pairing_code()
    wrong = "123456" if good != "123456" else "654321"
    codes = [_pair(intruder, wrong, "x").status_code for _ in range(edge_server.FAIL_MAX)]
    assert codes == [403] * edge_server.FAIL_MAX
    assert _pair(intruder, good, "x").status_code == 429                  # even the right code, once locked out
    other = TestClient(edge_server.app, client=("192.168.1.21", 50000))
    assert _pair(other, good, "sf-03").status_code == 200                 # per client address


def test_seed_idempotent_and_random_state_key(tmp_path, monkeypatch):
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    monkeypatch.delenv("DAYONE_BOX_KEY", raising=False)
    edge_server._vault = None; edge_server._registry = None
    c = TestClient(edge_server.app)
    seed = {"patients": [{"code": "2026-823-001", "facts": {}}]}
    c.post("/admin/seed", json=seed)
    assert c.post("/admin/seed", json=seed).json()["patients"] == 1
    assert len((tmp_path / "box.key").read_text()) == 64          # no public default key


def test_kept_image_role_based_access(tmp_path, monkeypatch):
    """Brief: keep the original page image linked to the record, with access by role (and every access logged).
    The midwife scope comes from the device token bound at pairing, not from a header the phone could forge."""
    monkeypatch.setattr(edge_server, "STATE", tmp_path)
    edge_server._vault = None; edge_server._registry = None
    box = TestClient(edge_server.app)
    img = "data:image/jpeg;base64,/9j/AAAA"
    body = {"record_id": "rec-a", "version": 2, "patient_id": "p1", "pages": [], "image": img,
            "midwife_id": "sf-01", "captured_at": "2026-10-04T09:00:00Z", "image_sha256": "ab"}
    assert box.post("/records", json=body).json()["duplicate"] is False
    phone = TestClient(edge_server.app, client=("192.168.1.20", 50000))
    t1 = _pair(phone, edge_server.pairing_code(), "sf-01").json()["token"]
    t2 = _pair(phone, edge_server.pairing_code(), "sf-02").json()["token"]
    ts = _pair(phone, edge_server.supervisor_code()).json()["token"]
    own = phone.get("/records/rec-a/image", headers={"X-DayOne-Token": t1})
    assert own.status_code == 200 and own.json()["image"] == img and own.json()["captured_at"].startswith("2026")
    spoof = phone.get("/records/rec-a/image", headers={"X-DayOne-Token": t2, "X-DayOne-Midwife": "sf-01"})
    assert spoof.status_code == 403                                       # the header is ignored
    assert phone.get("/records/rec-a/image", headers={"X-DayOne-Token": ts}).status_code == 200
    assert phone.get("/records/rec-a/image").status_code == 401
    log = edge_server._vault.load("access_log", [])
    assert [(e["role"], e["allowed"]) for e in log] == [("sage-femme", True), ("sage-femme", False), ("superviseure", True)]
    raw = b"".join(f.read_bytes() for f in tmp_path.glob("*.bin"))
    assert b"/9j/AAAA" not in raw                                    # image encrypted at rest
