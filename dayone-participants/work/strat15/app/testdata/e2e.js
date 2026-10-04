// Automated rehearsal of the demo (open http://localhost:8765/?db=e2e&e2e=1). Uses its own IndexedDB.
// Offline capture (+ blurry rejected) -> reload-safe queue -> connectivity back -> review with answers
// -> finish -> cross-checks -> match "Patiente 1" -> sync -> re-scan diff. Result in window.e2eResult.
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const checks = [];
  const ok = (name, cond, info = "") => { checks.push({ name, pass: !!cond, info }); };
  const blob = async n => (await fetch("testdata/" + n)).blob();
  try {
    await new Promise(res => { const r = indexedDB.deleteDatabase("e2e"); r.onsuccess = r.onerror = r.onblocked = res; });
    await fetch("/admin/seed", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patients: [
      { code: "2026-987-006", facts: { ddr: "21/11/2025", date_prevue: "28/08/2026" } }, { code: "2026-987-008", facts: { ddr: "02/02/2025" } }] }) });
    const before = await (await fetch("/records/stats")).json();
    await sleep(500);
    await app.unlock("2468");
    let wrong = false; try { await app.unlock("0000"); } catch { wrong = true; }
    ok("wrong PIN refused", wrong);
    app.setOffline(true);
    const pb = app.captureBlob(await blob("blurry.jpg"));
    for (let i = 0; i < 50 && !app.pendingButtons(); i++) await sleep(100);
    ok("quality gate asks for a retake", (app.pendingButtons() || []).includes("Reprendre la photo"));
    app.reply("Reprendre la photo");
    ok("blurry photo rejected", (await pb).rejected);
    for (const n of ["p1_cover.jpg", "p3_grossesse.jpg", "p4_accouchement.jpg"]) await app.captureBlob(await blob(n));
    let recs = await app.records();
    ok("3 pages queued offline", recs.length === 3 && recs.every(r => r.state === "EN_ATTENTE_IA"), recs.map(r => r.state).join(","));
    app.setOffline(false);
    // answer the agent like a midwife would, until the registry is finished and synced
    let finished = false, answered = 0, matchChoice = null;
    for (let t = 0; t < 900; t++) {
      const b = app.pendingButtons();
      if (b) {
        let a;
        if (b.includes("Patiente 1")) { a = "Patiente 1"; matchChoice = a; }
        else if (b.includes("Continuer")) a = "Continuer";
        else if (b.includes("Confirmer")) a = answered % 5 === 1 ? "Corriger" : "Confirmer";
        else if (b.includes("Saisir la valeur")) a = "Inconnu";
        else if (b.includes("Mettre à jour")) a = "Mettre à jour";
        else if (b.length) a = b.find(x => !/Reprendre/.test(x)) || b[0];
        else a = "12";                                           // free text (correction)
        app.reply(a); answered++;
      }
      recs = await app.records();
      if (!finished && recs.length === 3 && recs.every(r => r.state === "VALIDÉ")) { app.finish(); finished = true; }
      if (recs.length === 3 && recs.every(r => r.state === "SYNCHRONISÉ")) break;
      await sleep(200);
    }
    recs = await app.records();
    ok("all pages processed, reviewed, linked and synced", recs.every(r => r.state === "SYNCHRONISÉ"), recs.map(r => r.state).join(","));
    ok("one patient chosen by the midwife", matchChoice === "Patiente 1" && recs.every(r => r.patientId && r.patientId === recs[0].patientId));
    const after = await (await fetch("/records/stats")).json();
    ok("central store got the 3 records once", after.unique_records - before.unique_records === 3, JSON.stringify(after));
    const ev = await app.events();
    const path = ev.filter(e => e.id === recs[0].id).map(e => e.to);
    ok("lifecycle path is legal and complete", ["CAPTURÉ", "EN_ATTENTE_IA", "TRAITÉ_IA", "VALIDÉ", "PATIENTE_LIÉE", "ENREGISTRÉ", "SYNCHRONISÉ"].every(s => path.includes(s)), path.join(" → "));
    const res = await app.decryptResult(recs[0].id);
    ok("no identifier values in the record", !res.fields.some(f => /cin|telephone|adresse|parturiente|nom_du_mari/.test(f.key) && f.value));
    ok("agent asked questions with evidence crops", answered > 3 && document.querySelectorAll(".bot .bubble img").length > 0, `answers=${answered}`);
    const cached = [];
    for (const k of await caches.keys()) for (const r of await (await caches.open(k)).keys()) cached.push(new URL(r.url).pathname);
    ok("no patient data in the (unencrypted) browser cache", !cached.some(p => /^\/(records|process|session|match|dashboard|admin)/.test(p)), cached.join(","));
  } catch (e) { ok("no exception", false, e.stack || String(e)); }
  window.e2eResult = { passed: checks.every(c => c.pass), checks };
  const s = document.createElement("div"); s.className = "sys";
  s.innerHTML = "<span></span>"; s.firstChild.textContent = (window.e2eResult.passed ? "✅ E2E OK " : "❌ E2E FAILED ") + checks.map(c => (c.pass ? "✓ " : "✗ ") + c.name).join(" | ");
  document.getElementById("chat").appendChild(s);
})();
