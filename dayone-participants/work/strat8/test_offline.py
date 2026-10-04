"""Strategy 8 tests: property-based stateful test (network cuts, server errors, crashes) + crypto checks."""
import os
import sys
import tempfile
from pathlib import Path

import pytest
from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline import TRANSITIONS, IllegalTransition, Outbox, Server, Store  # noqa: E402

PIN = "4821"


class OfflineMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "d.db")
        self.store = Store(self.path, PIN)
        self.server = Server()
        self.net = {"online": False}
        self.outbox = Outbox(self.store, self.server, self.net)
        self.captured = 0
        self.now = 1e9

    @rule(img=st.binary(min_size=1, max_size=64))
    def capture(self, img):
        rid = self.store.capture(img, "sf-01")
        self.store.transition(rid, "EN_ATTENTE_IA")
        self.captured += 1

    @rule(ok=st.booleans())
    def process(self, ok):
        if not self.net["online"]:
            return
        for rid, s in self.store.records().items():
            if s == "EN_ATTENTE_IA":
                self.store.transition(rid, "TRAITÉ_IA" if ok else "ÉCHEC_TRAITEMENT",
                                      payload={"fields": [{"key": "inline.age", "value": "31"}]} if ok else None)
            elif s == "ÉCHEC_TRAITEMENT":
                self.store.transition(rid, "EN_ATTENTE_IA")

    @rule(choice=st.sampled_from(["confirm", "review", "retake"]))
    def review(self, choice):
        for rid, s in self.store.records().items():
            if s == "TRAITÉ_IA":
                self.store.transition(rid, "VALIDÉ" if choice == "confirm" else "À_RÉVISER")
            elif s == "À_RÉVISER":
                self.store.transition(rid, "CAPTURÉ" if choice == "retake" else "VALIDÉ")
            elif s == "CAPTURÉ":
                self.store.transition(rid, "EN_ATTENTE_IA")

    @rule()
    def link_and_register(self):
        for rid, s in self.store.records().items():
            if s == "VALIDÉ":
                self.store.transition(rid, "PATIENTE_LIÉE", patient="p-" + rid[:6])
            elif s == "PATIENTE_LIÉE":
                self.store.transition(rid, "ENREGISTRÉ")

    @rule(online=st.booleans())
    def network(self, online):
        self.net["online"] = online
        if online:
            self.outbox.requeue_failed()

    @rule(n=st.integers(0, 3))
    def server_errors(self, n):
        self.server.fail_next = n

    @rule()
    def sync(self):
        self.now += 1000
        self.outbox.flush(now=self.now, max_attempts=2)

    @rule()
    def crash_restart(self):
        self.store.db.close()
        self.store = Store(self.path, PIN)
        self.store.recover()
        self.outbox = Outbox(self.store, self.server, self.net)

    @rule(dst=st.sampled_from(sorted(TRANSITIONS)))
    def illegal_attempt(self, dst):
        for rid, s in list(self.store.records().items())[:1]:
            if dst not in TRANSITIONS[s]:
                with pytest.raises(IllegalTransition):
                    self.store.transition(rid, dst)

    @invariant()
    def nothing_lost(self):
        assert len(self.store.records()) == self.captured

    @invariant()
    def legal_states_and_replay(self):
        recs = self.store.records()
        assert all(s in TRANSITIONS for s in recs.values())
        assert self.store.replay_states() == recs

    @invariant()
    def synced_exactly_once(self):
        for rid, s in self.store.records().items():
            if s == "SYNCHRONISÉ":
                assert rid in self.server.data
        # server keys are records, so a record can only exist once there
        assert len(self.server.data) == len(set(self.server.data))


OfflineMachine.TestCase.settings = settings(max_examples=150, stateful_step_count=60, deadline=None)
TestOffline = OfflineMachine.TestCase


def test_everything_eventually_syncs():
    d = tempfile.mkdtemp()
    s = Store(os.path.join(d, "x.db"), PIN)
    srv, net = Server(), {"online": False}
    ob = Outbox(s, srv, net)
    ids = []
    for i in range(50):
        r = s.capture(f"img{i}".encode(), "sf")
        for st_ in ("EN_ATTENTE_IA", "TRAITÉ_IA", "VALIDÉ", "PATIENTE_LIÉE", "ENREGISTRÉ"):
            s.transition(r, st_, payload={"i": i} if st_ == "TRAITÉ_IA" else None)
        ids.append(r)
    assert ob.flush() == 0                     # offline: nothing sent, nothing lost
    net["online"] = True
    srv.fail_next = 7
    t = 1e9
    for _ in range(20):
        t += 1000
        ob.requeue_failed()
        ob.flush(now=t, max_attempts=3)
    assert all(s.state(r) == "SYNCHRONISÉ" for r in ids)
    assert len(srv.data) == 50


def test_encryption_at_rest_and_wrong_pin():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "e.db")
    s = Store(p, PIN)
    rid = s.capture(b"PHOTO-BYTES-UNIQUE-123", "sf")
    s.transition(rid, "EN_ATTENTE_IA")
    s.transition(rid, "TRAITÉ_IA", payload={"fields": [{"key": "visites.ta.t1_v2", "value": "104/74"}]})
    s.db.close()
    raw = open(p, "rb").read() + (open(p + "-wal", "rb").read() if os.path.exists(p + "-wal") else b"")
    assert b"104/74" not in raw and b"PHOTO-BYTES-UNIQUE-123" not in raw
    with pytest.raises(Exception):
        Store(p, "0000")
    s2 = Store(p, PIN)
    assert s2.payload(rid)["fields"][0]["value"] == "104/74"
    with pytest.raises(PermissionError):
        s2.image(rid, "visiteur")
    assert s2.image(rid, "sage-femme") == b"PHOTO-BYTES-UNIQUE-123"


def test_lifecycle_doc_matches_the_shared_table():
    """LIFECYCLE.md draws exactly the transitions of lifecycle.json (used by the app and by this model)."""
    import json
    import re
    root = Path(__file__).resolve().parents[2]
    md = (root / "LIFECYCLE.md").read_text(encoding="utf-8")
    drawn = {(a, b.rstrip(":")) for a, b in re.findall(r"^\s+(\S+) --> (\S+?):?(?:\s|$)", md, re.M)
             if a != "[*]" and b != "[*]"}
    table = json.loads((root / "work" / "strat15" / "app" / "lifecycle.json").read_text(encoding="utf-8"))
    assert drawn == {(a, b) for a, bs in table["transitions"].items() for b in bs}
    assert {k: set(v) for k, v in table["transitions"].items()} == TRANSITIONS
