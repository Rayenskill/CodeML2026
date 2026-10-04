# Demo script (≈ 3 minutes)

Shows what the brief asks for: **an offline capture, the return of connectivity, the review of an uncertain
field and a patient-match decision** — plus the quality gate, re-scan diff and privacy.

## Setup (once, before the jury)

```bash
./run_box.sh            # or .\run_box.ps1 on Windows — edge box + phone app on port 8765
```

* **On the box's own browser** open `http://localhost:8765` (a secure origin: no certificate needed).
* **On a phone** start the box with `./run_box.sh 8765 --https` (`.\run_box.ps1 -Https`), open
  `https://<box-ip>:8765` and accept the self-signed certificate once (or use mkcert, see
  `work/strat20/make_cert.py`): the phone's encryption (WebCrypto) only works over HTTPS. At the first network
  call the agent asks for the **6-digit pairing code printed by the box at start-up** — type it in the chat; the
  phone gets its own device token. (The box also prints a supervisor code: pairing with it gives the supervisor
  role.) To reopen the app offline away from the box, use a mkcert certificate instead of the self-signed one.

Seed two known patients so the match step has candidates (simulates earlier visits; idempotent, safe to re-run):

```bash
curl -X POST http://localhost:8765/admin/seed -H "Content-Type: application/json" -d "{\"patients\":[{\"code\":\"2026-987-006\",\"facts\":{\"ddr\":\"21/11/2025\",\"date_prevue\":\"28/08/2026\"}},{\"code\":\"2026-987-008\",\"facts\":{\"ddr\":\"02/02/2025\"}}]}"
```

Demo photos (synthetic specimen pages photographed with phone-like degradations) are in
`work/strat15/app/testdata/`: `p1_cover.jpg`, `p3_grossesse.jpg`, `p4_accouchement.jpg`, `blurry.jpg`.

## Script

| # | Action | What the jury sees | Rubric |
|---|---|---|---|
| 1 | Unlock with the PIN | records are encrypted with a key derived from the PIN; a wrong PIN is refused | Privacy |
| 2 | Tap 📶 → 📴 (offline) | header says *hors ligne* | Offline |
| 3 | 📷 send `blurry.jpg` | "La photo est floue…" → *Reprendre la photo* (on-device quality gate) | Bonus |
| 4 | 📷 send the 3 pages | each: "📥 Page reçue et chiffrée… en attente de traitement IA" ; ☰ shows 3 × EN_ATTENTE_IA | Offline |
| 5 | Reload the page / close the app, unlock again | "3 dossier(s) en cours repris": nothing lost | Offline |
| 6 | Tap 📴 → 📶 (online) | "📶 La connexion est revenue…", pages go TRAITÉ_IA | Offline |
| 7 | Review | "📄 Page « Accouchement » lue : 29 champs sûrs, 1 à vérifier…" then a question **with the image crop**: "Je ne suis pas sûre de « … » : j'ai lu « … » (62 %)" → *Corriger* and type the value; on another one *Confirmer*; on an illegible one *Saisir la valeur* / *Inconnu* | Uncertainty, Review |
| 8 | (long page) | after 8 questions: "Il reste N champs incertains" → *Continuer* or *Valider, garder « à réviser »* (doubt stays recorded) | Review |
| 9 | *Terminer le registre* | "🔎 Vérification croisée" (DDR + 280 j, ages, same fact on two pages, visit table): every value a rule changed or flagged is shown with both readings — *Prendre « … »* / *Garder « … »* | Extraction |
| 10 | Match | "Code lu sur la fiche : 2026-987-006" (a code not read with certainty is confirmed or typed first: the linking key is never used unconfirmed) → *Patiente 1 · DDR 21/11/2025 ✓* / *Aucune, créer* / *Je ne sais pas* → choose **Patiente 1** | Linking |
| 11 | Sync | "☁️ Synchronisé (3 page(s))"; ☰ shows SYNCHRONISÉ | Offline |
| 12 | Send `p3_grossesse.jpg` again, finish | "📚 Page déjà numérisée…" diff → *Mettre à jour* / *Garder l'ancien* (re-digitisation) | Review |
| 12b | On the box: `curl http://localhost:8765/records/<record-id>/image` (full record id: `app.records()` in the browser console) | the kept image of the record, identity zones masked. A phone paired for another midwife gets 403, a device paired with the supervisor code gets it; every access is logged | Privacy |
| 13 | Open `http://<box>:8765/dashboard` | anonymised aggregates from the synced records, cells < 5 suppressed, missing shown | Bonus |
| 14 | Toggle **EN** | the same agent in English | Bonus |

Also worth showing: a real booklet photo (`data/Paper Registry/1-3.jpg`) → "🤔 Je ne reconnais pas cette page…
Je préfère ne rien inventer" (*Reprendre la photo* / *Saisie manuelle*): the agent never hides its doubt. As the
page could not be registered, its identifier zones cannot be located: the box sends no image crop and the phone
deletes the photo (nothing unmasked is kept).

## Automated rehearsal

Open `http://localhost:8765/?db=e2e&e2e=1` on the box: the whole flow above runs by itself in a separate
IndexedDB and ends with ✅/❌ per check (also in `window.e2eResult`). The Python side is covered by `pytest work`.
