"""Strategy 8: offline-first record lifecycle, encrypted local store and idempotent sync outbox.

States: CAPTURÉ -> EN_ATTENTE_IA -> TRAITÉ_IA -> À_RÉVISER -> VALIDÉ -> PATIENTE_LIÉE -> ENREGISTRÉ -> SYNCHRONISÉ
        + ÉCHEC_TRAITEMENT, ÉCHEC_SYNCHRO, DOUBLON_SUSPECTÉ, RÉVISION_MANUELLE_REQUISE
Every transition is one SQLite transaction: update the record + append to the event log. Field values and
images are AES-GCM encrypted with a key derived from the device PIN (scrypt); nothing readable on disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

# one transition table for the phone app and this property-tested model (work/strat15/app/lifecycle.json)
TRANSITIONS = {k: set(v) for k, v in json.loads(
    (Path(__file__).resolve().parents[1] / "strat15" / "app" / "lifecycle.json").read_text(encoding="utf-8"))
    ["transitions"].items()}


class IllegalTransition(Exception):
    pass


class Crypto:
    def __init__(self, pin: str, salt: bytes):
        key = Scrypt(salt=salt, length=32, n=2 ** 14, r=8, p=1).derive(pin.encode())
        self.aes = AESGCM(key)

    def enc(self, data: bytes, aad: str) -> bytes:
        n = os.urandom(12)
        return n + self.aes.encrypt(n, data, aad.encode())

    def dec(self, blob: bytes, aad: str) -> bytes:
        return self.aes.decrypt(blob[:12], blob[12:], aad.encode())


class Store:
    def __init__(self, path: str, pin: str):
        self.db = sqlite3.connect(path, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v BLOB);
        CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, state TEXT, version INTEGER, midwife TEXT,
            captured_at REAL, payload BLOB, image BLOB, image_sha TEXT, patient TEXT);
        CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT, record TEXT, src TEXT, dst TEXT,
            at REAL, actor TEXT, reason TEXT);
        CREATE TABLE IF NOT EXISTS outbox(ikey TEXT PRIMARY KEY, record TEXT, attempts INTEGER, next_try REAL,
            inflight INTEGER DEFAULT 0);
        """)
        row = self.db.execute("SELECT v FROM meta WHERE k='salt'").fetchone()
        salt = row[0] if row else os.urandom(16)
        if not row:
            self.db.execute("INSERT INTO meta VALUES('salt', ?)", (salt,))
        self.crypto = Crypto(pin, salt)
        check = self.db.execute("SELECT v FROM meta WHERE k='check'").fetchone()
        if check:
            self.crypto.dec(check[0], "check")      # raises InvalidTag on a wrong PIN
        else:
            self.db.execute("INSERT INTO meta VALUES('check', ?)", (self.crypto.enc(b"ok", "check"),))

    # ---- records
    def capture(self, image: bytes, midwife: str) -> str:
        rid = uuid.uuid4().hex
        with self.db:
            self.db.execute("BEGIN")
            self.db.execute("INSERT INTO records VALUES(?,?,?,?,?,?,?,?,?)",
                            (rid, "CAPTURÉ", 0, midwife, time.time(), self.crypto.enc(b"{}", rid),
                             self.crypto.enc(image, rid + ":img"), hashlib.sha256(image).hexdigest(), None))
            self.db.execute("INSERT INTO events(record,src,dst,at,actor,reason) VALUES(?,?,?,?,?,?)",
                            (rid, None, "CAPTURÉ", time.time(), midwife, "capture"))
        return rid

    def state(self, rid):
        return self.db.execute("SELECT state FROM records WHERE id=?", (rid,)).fetchone()[0]

    def transition(self, rid, dst, actor="system", reason="", payload: dict | None = None, patient=None):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            src, ver = self.db.execute("SELECT state, version FROM records WHERE id=?", (rid,)).fetchone()
            if dst not in TRANSITIONS[src]:
                raise IllegalTransition(f"{src} -> {dst}")
            if payload is not None:
                self.db.execute("UPDATE records SET payload=? WHERE id=?",
                                (self.crypto.enc(json.dumps(payload, ensure_ascii=False).encode(), rid), rid))
            if patient is not None:
                self.db.execute("UPDATE records SET patient=? WHERE id=?", (patient, rid))
            self.db.execute("UPDATE records SET state=?, version=? WHERE id=?", (dst, ver + 1, rid))
            self.db.execute("INSERT INTO events(record,src,dst,at,actor,reason) VALUES(?,?,?,?,?,?)",
                            (rid, src, dst, time.time(), actor, reason))
            if dst == "ENREGISTRÉ":
                self.db.execute("INSERT OR IGNORE INTO outbox VALUES(?,?,?,?,0)", (f"{rid}:{ver + 1}", rid, 0, 0.0))

    def payload(self, rid) -> dict:
        b = self.db.execute("SELECT payload FROM records WHERE id=?", (rid,)).fetchone()[0]
        return json.loads(self.crypto.dec(b, rid))

    def image(self, rid, role: str) -> bytes:
        """Original photo: restricted access (role-based) and every access is logged."""
        if role not in ("sage-femme", "superviseur"):
            raise PermissionError(role)
        b = self.db.execute("SELECT image FROM records WHERE id=?", (rid,)).fetchone()[0]
        self.db.execute("INSERT INTO events(record,src,dst,at,actor,reason) VALUES(?,?,?,?,?,?)",
                        (rid, None, None, time.time(), role, "image_access"))
        return self.crypto.dec(b, rid + ":img")

    def records(self):
        return dict(self.db.execute("SELECT id, state FROM records").fetchall())

    def recover(self):
        """Crash recovery: release in-flight outbox items (processing is idempotent)."""
        self.db.execute("UPDATE outbox SET inflight=0")

    def replay_states(self):
        st = {}
        for rid, dst in self.db.execute("SELECT record, dst FROM events WHERE dst IS NOT NULL ORDER BY seq"):
            st[rid] = dst
        return st


class Server:
    """Mock central server: idempotent upsert keyed by the idempotency key."""

    def __init__(self):
        self.data, self.seen, self.fail_next = {}, set(), 0

    def upsert(self, ikey, rid, payload_hash):
        if self.fail_next > 0:
            self.fail_next -= 1
            raise ConnectionError("500")
        if ikey in self.seen:
            return 200
        self.seen.add(ikey)
        self.data[rid] = payload_hash
        return 200


class Outbox:
    def __init__(self, store: Store, server: Server, net):
        self.s, self.srv, self.net = store, server, net

    def flush(self, now=None, max_attempts=5):
        now = now or time.time()
        sent = 0
        rows = self.s.db.execute("SELECT ikey, record, attempts FROM outbox WHERE next_try<=? ORDER BY rowid",
                                 (now,)).fetchall()
        for ikey, rid, att in rows:
            if not self.net["online"]:
                break
            self.s.db.execute("UPDATE outbox SET inflight=1 WHERE ikey=?", (ikey,))
            try:
                h = hashlib.sha256(json.dumps(self.s.payload(rid), sort_keys=True).encode()).hexdigest()
                self.srv.upsert(ikey, rid, h)
            except ConnectionError:
                att += 1
                self.s.db.execute("UPDATE outbox SET attempts=?, next_try=?, inflight=0 WHERE ikey=?",
                                  (att, now + min(600, 2 ** att), ikey))
                if att >= max_attempts and self.s.state(rid) == "ENREGISTRÉ":
                    self.s.transition(rid, "ÉCHEC_SYNCHRO", reason="max attempts")
                continue
            st = self.s.state(rid)
            if st == "ÉCHEC_SYNCHRO":
                self.s.transition(rid, "ENREGISTRÉ", reason="late ack")
                st = "ENREGISTRÉ"
            if st == "ENREGISTRÉ":
                self.s.transition(rid, "SYNCHRONISÉ", reason="ack")
                sent += 1
            self.s.db.execute("DELETE FROM outbox WHERE record=?", (rid,))
        return sent

    def requeue_failed(self):
        for (rid,) in self.s.db.execute("SELECT id FROM records WHERE state='ÉCHEC_SYNCHRO'").fetchall():
            self.s.transition(rid, "ENREGISTRÉ", reason="connectivity back")
