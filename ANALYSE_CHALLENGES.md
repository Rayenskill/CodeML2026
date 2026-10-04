# CodeML 2026 — Which challenge should we pick? (full analysis)

> **Who this is for:** everyone on the team, including people who were not there when the analysis was done.
> **Reading time:** 10 min for sections 1–5, 30–40 min for everything.
> **Date of analysis:** 2026-10-03. Based on the official instructions (`consignes*.pdf`) and the data actually present in each `*-participants/` folder.
> **Confidentiality:** the raw data of JADCO, L2C and CorroborAI is confidential and is **not** in this repo (git-ignored). This file only quotes aggregate statistics from those datasets. The other challenges' data (fictional or synthetic) is included. See [Appendix C](#appendix-c--confidentiality-rules-per-challenge).
>
> Notation: **[verified]** = checked directly in the files. **[hypothesis]** = inference that still needs confirming.

---

## Table of contents

1. [Context: what we were trying to decide](#1-context-what-we-were-trying-to-decide)
2. [TL;DR: ranking and recommendation](#2-tldr-ranking-and-recommendation)
3. [What changed compared with our initial ranking](#3-what-changed-compared-with-our-initial-ranking)
4. [Questions to ask the organizers before committing](#4-questions-to-ask-the-organizers-before-committing)
5. [How the analysis was done](#5-how-the-analysis-was-done)
6. [The key insight: read the rubric, not the ambition](#6-the-key-insight-read-the-rubric-not-the-ambition)
7. [Challenge-by-challenge analysis](#7-challenge-by-challenge-analysis)
   - 7.1 [NOVA / Projet 360 (Loto-Québec): recommended](#71-nova--projet-360-loto-québec--recommended)
   - 7.2 [JADCO / Collection Équinoxe](#72-jadco--collection-équinoxe)
   - 7.3 [IVADO / EquiAlgo](#73-ivado--equialgo)
   - 7.4 [CorroborAI (Loto-Québec)](#74-corroborai-loto-québec)
   - 7.5 [DayOne / Offline Midwife](#75-dayone--offline-midwife)
   - 7.6 [SN-SF / OptiFrame](#76-sn-sf--optiframe)
   - 7.7 [L2C / plans vs shop drawings](#77-l2c--plans-vs-shop-drawings)
   - 7.8 [Propolys / Security & AI pitch](#78-propolys--security--ai-pitch)
8. [Comparison matrix](#8-comparison-matrix)
9. [Final recommendation and decision tree](#9-final-recommendation-and-decision-tree)
10. [Glossary](#10-glossary)
- [Appendix A: reproduce the probes](#appendix-a--reproduce-the-probes)
- [Appendix B: file inventory per folder](#appendix-b--file-inventory-per-folder)
- [Appendix C: confidentiality rules per challenge](#appendix-c--confidentiality-rules-per-challenge)

---

## 1. Context: what we were trying to decide

- 24-hour hackathon, 8 challenges from different partners. We pick **one** (unless multi-entry is allowed, see §4).
- Our strengths: Python, ML, time series and forecasting, quant analysis, backtesting, optimisation, LLMs and agents, structured extraction, scoring systems, graphs, data pipelines, software dev.
- **Goal:** maximise the jury's score, not build the most impressive thing. We want a solution that is complete, robust, easy to explain, aligned with the rubric, and demoable at hour 24.

---

## 2. TL;DR: ranking and recommendation

| Rank | Challenge | Expected score /100 | Execution risk | One-line reason |
|---|---|---|---|---|
| **1** | **NOVA (Loto-Québec, Projet 360)** | **80–92** | **Low** | Half the points are 10 questions known in advance, over a tiny corpus we read entirely. All 10 answers are already drafted with sources (§7.1.4). |
| 2 | JADCO (rent growth 2026) | 75–88 | Low | Best fit with our forecasting skills, and method is graded. But the starter notebook already gives ~50 pts to every team. |
| 3 | IVADO EquiAlgo (fair scholarships) | 65–85 | Medium | 35 pts are auto-scored against a hidden reference. We found and tested a lever (region-neutralised model) that should beat what most teams will do. |
| 4 | CorroborAI (HR data reconciliation) | 55–70 | Medium | ~23 employees, rules mostly deterministic, 35 pts on a hidden test set, and 25 pts for "AI relevance" that are hard to earn honestly. |
| 5 | DayOne (paper registry → digital) | 45–65 | High | 30 pts on handwriting extraction (FR/AR/EN); the CSV provided is **not** page-level ground truth; 5 subsystems to build. |
| 6 | OptiFrame (lens → 3D-printed frame) | 25–75 | Very high | 30 pts = ≤1 mm accuracy measured live by the jury with our physical setup; web app + vision + CAD. |
| 7 | Propolys (startup pitch) | subjective | Low (technical) | 3-min pitch, $400 prize. Only as a side quest if allowed. |
| 8 | L2C (rebar plans vs shop drawings) | 15–45 | Extreme | 87% of shop drawings have **no text layer**, sheets up to 48"×36", cloud AI forbidden. |

**Recommendation:** **NOVA** for maximum expected score at minimum risk. **JADCO** if we prefer to use our quant skills and accept less differentiation. **EquiAlgo** if prizes are per challenge and NOVA/JADCO look crowded.
The score ranges are **estimates**, not measurements.

---

## 3. What changed compared with our initial ranking

Our pre-analysis (before seeing data) ranked: JADCO > NOVA > CorroborAI > IVADO > DayOne > SNSF > L2C > Propolys.

| Challenge | Before | After | Why it moved |
|---|---|---|---|
| NOVA | 2 | **1** | The README inside the zip lists the **10 graded questions** in advance (50 pts). The corpus is 64 small files we could read entirely; all answers found in ~20 min. The other 50 pts are an explicit checklist. |
| JADCO | 1 | 2 | Still excellent, but the **starter notebook already implements** the mix effect, same-unit pairing, concessions comparison and renewal split. Those points are a floor for everyone. |
| EquiAlgo | 4 | 3 | We tested an approach that should clearly beat the default tool suggested in the brief, on the **objectively scored** part. |
| CorroborAI | 3 | 4 | Data is tiny (23 vs 22 rows) and messy; some differences might be anonymisation artefacts; the "AI relevance" criterion is 25 pts. |
| DayOne | 5 | 5 | Confirmed hard: the provided CSV does not correspond to the images. |
| L2C | 7 | 8 | Confirmed worst: shop drawings have no extractable text. |

Corrections to assumptions in our pre-analysis:
- **EquiAlgo** is not "predict merit" in a free sense. The grant rate on the 4,000 candidates **must be 36–44%** or the technical section scores 0. Scoring is against a **hidden reference**, not the historical decisions.
- **NOVA**'s exact rubric is in `README.txt` inside `NOVA_ETUDIANTS.zip`, not in the PDF.
- **OptiFrame** (SN-SF = Santé Numérique Sans Frontières): 30 of the 100 pts are pure measurement accuracy, tested live.

---

## 4. Questions to ask the organizers before committing

These can change the decision:

1. **Are prizes awarded per challenge** (we compete only against teams on the same challenge)? If yes, popularity matters: a crowded easy challenge becomes a coin flip among the top teams.
2. **Can a team submit to more than one challenge?** If yes, NOVA (~25–35 person-hours) plus JADCO in parallel is realistic; Propolys could be a 3-hour side quest.
3. **EquiAlgo:** is the HxBuddy leaderboard F1/accuracy computed **against the hidden reference**? How many submissions are allowed? If yes, each submission tells us something about the reference.
4. **NOVA:** in what form will the "new event" arrive during the final presentation (document, email, verbal)? How much time do we get to integrate it?

---

## 5. How the analysis was done

So you can trust (or re-check) the conclusions:

1. **Read every instruction PDF.** Text extracted with `pdftotext -layout`. In particular, the scoring grids (barèmes).
2. **Opened every dataset.**
   - CSVs with pandas.
   - Excel files with `openpyxl`. Note: on this machine pandas 3.0.2 refuses openpyxl 3.1.2 (`ImportError: Pandas requires version '3.1.5'`). Either `pip install -U openpyxl` in a venv, or read with openpyxl directly as we did.
   - Emails (`.eml`) with Python's `email` package, listing attachments.
   - Images opened and looked at (NOVA screenshots, DayOne registry photos).
3. **Ran quick probes** to test feasibility, not to build solutions:
   - JADCO: same-unit growth by year × renewal × province; check of the `stelz1`/`stelz3` trap.
   - EquiAlgo: logistic regression on the committee decisions, then a region-neutralised counterfactual.
   - L2C: counted extractable words in all 137 shop-drawing PDFs.
   - NOVA: read the whole corpus and drafted the 10 answers.
4. **Scored each challenge against its own rubric:** which points are easy, which are hard, what depends on data we cannot see.

All probe code is in [Appendix A](#appendix-a--reproduce-the-probes) so anyone can rerun it.

---

## 6. The key insight: read the rubric, not the ambition

Three rubrics explicitly say sophistication does not earn points:

- **NOVA:** *"Une belle interface, une technologie particulière ou le seul recours à l'IA ne donnent pas de points supplémentaires. La précision, les preuves et l'utilité priment."*
- **JADCO:** *"La méthode compte plus que la proximité du résultat."* Points are **lost** for magic numbers, inconsistent naming, matching on `sSite+sUnitCode`, missing Markdown.
- **CorroborAI:** *"une approche hybride règles + IA sera souvent plus fiable, plus explicable et plus facile à valider."*

So the winning posture is **precision, traceability and explicit method**, not a big architecture. That favours NOVA and JADCO, where careful work translates directly into points.

---

## 7. Challenge-by-challenge analysis

Each section follows the same template: problem → scoring → what's in the data → findings → risks → plan → what not to build → variants → verdict.

---

### 7.1 NOVA / Projet 360 (Loto-Québec): recommended

#### 7.1.1 The problem
A fictional IT project ("NOVA", a request-tracking portal built by the vendor **Boréal Numérique**) has its information scattered across emails, meeting notes, tickets, contracts, invoices, plans and Teams chats. Some information is outdated or contradictory. Build an **"operational memory"** that lets someone take over the project: find evidence, answer questions, know what is currently valid, and update the state when a new event arrives **without erasing history**.
The reference date is **30 Sept 2026, 09:00 Montréal time**. During the final presentation, a **new piece of information** is given and must be integrated live.

#### 7.1.2 Official scoring (from `README.txt` inside the zip) [verified]
| Criterion | Points | How it is judged |
|---|---|---|
| 10 factual answers (Q01–Q10) | **50** | Each 0, 3 or 5 for **accuracy and nuance** |
| Evidence & navigation | 10 | 5: ≥3 answers with file + precise locator. 5: ≥2 of them cross distinct sources. |
| Chronology & contradictions | 10 | 5: distinguish proposal / decision / validation with dates and sources. 5: explain ≥2 contradictions by authority or date, **one in a plan or risk register**. |
| Brief & actions | 10 | 5: 1-page brief covering the 5 themes. 5: link the 3 go-live conditions to actions, owners, due dates (known or "à confirmer"). |
| Use & uncertainty | 10 | 5: jury can open the deliverable and find a proof. 5: team shows its search and states limits during the follow-up question. |
| Update after the event | 10 | 5: distinguish problem status / prior decision / new proposal. 5: keep the baseline; sourced impacts and actions **without inventing approvals or closing other conditions**. |

Required deliverables: (1) 1-page takeover brief (owner, approved date and conditions, scope, budget, invoices status, priorities); (2) searchable memory (timeline, decisions, resolved contradictions, sources, remaining actions, each with owner/evidence/due date, distinguishing **our recommendation** from a **documented commitment**); (3) the 10 answers, each with a file and precise locator; (4) the post-event updated state with diff, keeping the original; (5) a short user guide (how to open, navigate, tools used, manual steps, limits).
Any technology is allowed, including AI tools (must be declared). No chatbot required. **No paid subscription needed by the jury** to view the deliverable.

#### 7.1.3 What's in the data [verified]
64 files, ~700 KB, all fictional (so external LLMs are allowed):

| Folder | Content |
|---|---|
| `01_Courriels` | 12 emails (E01–E12), some with PDF attachments that duplicate files elsewhere |
| `02_Reunions` | 6 meeting minutes / transcripts (7 Jul → 26 Sept) |
| `03_Tickets` | 8 tickets (ACC-301/302/303, DATA-401, INT-101, OPS-601, PERF-501, SEC-210) + screenshots, a log extract, a CSV sample |
| `04_Documents_projet` | Charter v1, transition note, Plan v2, Plan v3 (12 Sept), status report 21 Sept, risk register 29 Sept |
| `05_Contrats_et_finances` | Contract, CR-01 (approved), CR-04 (draft), invoices INV-001/002/003 |
| `06_Architecture_et_decisions` | ADR-007, Architecture v1 and v2, phase-2 scope decision |
| `07_Conversations_Teams` | 4 short chat logs |
| `08_Archives_et_documents_connexes` | Distractors: duplicate email, another project's invoice (ORION), newsletter, Excel training invite, anonymous personal notes, June preliminary plan |

Everything is text-extractable (pdftotext, openpyxl, email parsing). The 8 PNG screenshots must be looked at; one of them holds a graded detail (Q10).

#### 7.1.4 Draft answers to the 10 questions (to be re-verified by a second person)
| Q | Question (short) | Draft answer | Sources (with locators) |
|---|---|---|---|
| Q01 | Approved go-live date and caveat? | **22 Oct 2026**, approved but **conditional**: not an automatic go. Three go-live conditions remain (see Q10). | M04 transcript 15:22–15:28; M06 10:09–10:15; E09 (27 Sept); M05 |
| Q02 | Why did the date change; status of the original cause? | The internal connector problem (**INT-101**: HTTP 401 errors after a secret change, expired service token, then intermittent errors) consumed time and became the critical path. **Current status: resolved.** Fix deployed and validated 17 Sept (120/120 searches OK), ticket closed. The date was **not** moved back; no decision to do so. Risk register R-01 is stale on this. | E05; M04 15:02–15:06; INT-101 ticket + `INT-101_extrait_logs.txt`; E12; M05; Registre_Risques R-01 |
| Q03 | Who approved, when; proposal vs approval? | **Proposed** by Julien Moreau (Boréal) by email on 8 Sept ("il s'agit d'une proposition de notre part", E05), repeated at the 10 Sept committee. **Approved** on 10 Sept 2026 by the steering committee: decision formulated by Élodie Caron (project manager at the time), no objection from Sophie, Marc, Olivier, Nicolas. | E05; M04 15:02, 15:22–15:25; Teams 15 Sept; transition note |
| Q04 | Who is project manager and since when? | **Nicolas Perron since 16 Sept 2026** (Élodie Caron before, since 7 July). Plan v3 (12 Sept) already names Nicolas on P-06; Plan v2 still names Élodie. | E06; `Note_transition_Elodie_16sept.txt`; Teams 16 Sept; M01; Charter |
| Q05 | Authorised contract amount and how it's computed? | **204,000 $** = 180,000 $ initial maximum (contract) + 24,000 $ CR-01 approved 14 Aug by the project committee. CR-04 (18,000 $) is a **draft**, so not included. Context: invoiced 186k (INV-001 60k paid, INV-002 72k paid incl. CR-01, INV-003 54k in validation); paid 132k. INV-778 (41k) belongs to project ORION, so excluded. | Contract p.1; CR-01 p.1; CR-04 p.1; INV-001/002/003 |
| Q06 | What's wrong with INV-003; amount; treatment? | INV-003 has an **18,000 $** line "Optimisation interface mobile – CR-04", but CR-04 was never approved: still a draft, "aucune approbation" on 10 Sept, deferred to phase 2 on 24 Sept, "Facturer du CR-04, non" on 26 Sept. The contract requires a written, approved change **before execution and invoicing**. Treatment: **do not release the 18k line**; process the 36k milestone-3 line through normal validation; ask Boréal for a corrected invoice or credit note. Mobile work only after a new formal approval (phase 2). | INV-003; E07 (Amélie Fortin, Finance); CR-04; M04 15:34–15:40; `Decision_Portee_Phase2.md`; E10; M06 10:24–10:27; Teams 22 Sept; contract "Gestion des changements" |
| Q07 | Where must production data be hosted; proof of implementation? | **Canada Central** (ADR-007, accepted, decided 23 July). Implementation: Boréal declares migration complete on 26 Aug (E03) with Architecture v2 (25 Aug); **verified by the architecture team** per the 27 Aug committee notes. E03's attachment *is* the Arch v2 file, so it is not an independent confirmation. | M02 09:06–09:12; ADR-007; E03; `Architecture_NOVA_v2.pdf`; M03 "Résumé de gestion" |
| Q08 | Is security accepted? Delivery vs validation | **No.** SEC-210 fix **delivered/deployed** on 19 Sept by Boréal. Security **validation pending**: ticket status EN VALIDATION, re-test planned (26 Sept). The status report "Sécurité VERT" and Alex's draft message "sécurité complétée" are wrong. | SEC-210 ticket (19 & 26 Sept comments); E08; Teams 19 Sept; M06 10:01–10:03; status report 21 Sept; E11 |
| Q09 | Is accessibility complete; what remains? | **No.** ACC-301 (labels) and ACC-302 (contrast 2.1:1 → 5.3:1) closed and validated on 15 and 20 Aug. **ACC-303 OPEN**, high priority: keyboard focus trapped in the modal between "Nom" and "Commentaire", so the **"Enregistrer" button is unreachable with Tab**. Blocking per Mélissa (26 Sept); fix only "announced for the next build". | ACC-301/302/303 tickets; M03; M05; M06 10:05; E04; `ACC-303_focus.png` |
| Q10 | Three go-live conditions; missing runbook work from the screenshot? | (1) **Security validation of SEC-210**; (2) **closure of ACC-303**; (3) **approval of the runbook including rollback**. The runbook screenshot (version of 25 Sept) shows **step 4 "Procédure de retour arrière" = TODO** and **step 5 "Validation fonctionnelle post-déploiement" = À compléter** (steps 1–3 OK). Olivier still had no final version on 29 Sept. | M06 10:07–10:15; OPS-601 ticket (25/26/29 Sept); `OPS-601_runbook.png`; E09 |

#### 7.1.5 Traps planted in the corpus (where careless teams get 3 instead of 5)
- **Proposal ≠ decision.** Boréal *proposed* 22 Oct (8 Sept). The committee *approved* it (10 Sept).
- **Delivered ≠ accepted.** SEC-210 "fix deployed" ≠ security acceptance (Teams 19 Sept: *"déployé != accepté"*).
- **Official-looking but stale documents.** Plan v3 (12 Sept) still says 15 Oct. The 29 Sept risk register still lists the connector risk as open, "suivi au 9 septembre". The 21 Sept status report shows security and accessibility "VERT".
- **A graded detail only visible in an image.** The OPS-601 text says "rollback + another step"; only the PNG shows step 5.
- **Money.** Authorised = 180k + 24k = **204k**, not 180k. Separate *authorised*, *invoiced* and *paid*. Ignore the ORION invoice.
- **Duplicates are not independent confirmations.** `Courriel_archive_17sept.eml` = E12. Email attachments duplicate the standalone PDFs.
- **Noise.** Anonymous personal notes ("15 oct encore date? probablement"), newsletter, Excel training, an HDMI cable, room 18B.

#### 7.1.6 Contradictions ready for the "chronology" criterion
| # | Claim A | Claim B | Resolution |
|---|---|---|---|
| 1 | **Plan v3 (12 Sept): go-live 15 Oct** *(a plan ✔)* | Committee 10 Sept: **22 Oct approved** | Committee has higher authority and decided earlier; Teams 15 Sept: *"le plan projet n'a visiblement pas encore été corrigé"*. |
| 2 | **Risk register 29 Sept: R-01 connector risk "Ouvert"** *(a register ✔)* | INT-101 closed and validated 17 Sept | The register entry says "Suivi au 9 septembre", so it is stale despite the file date. |
| 3 | Status report 21 Sept: security & accessibility **VERT** | SEC-210 EN VALIDATION; ACC-303 OPEN | The report itself says it predates detailed ticket checks. |
| 4 | Julien (22 Sept): advanced mobile was in initial scope | Charter, contract, M01 note, decision of 24 Sept | Contract scope + formal decision override a vendor's belief. |

#### 7.1.7 Proposed architecture (deliberately simple)
```
corpus/  (original files, untouched, shipped with the deliverable)
   │  scripted extraction: .eml → text + attachments, pdftotext, openpyxl (incl. cell comments),
   │  manual reading / vision for the 8 PNG screenshots
   ▼
facts.yaml   ← hand-curated, LLM-cross-checked, ~60–90 entries. Each fact:
   id, statement, type {fact | proposal | decision | validation | action | risk | contradiction},
   status, date_effective, source_file, locator (page / cell / timestamp / line / screenshot),
   authority (committee > PM > ticket owner > vendor > status report > personal note),
   supersedes[], superseded_by[], valid_from, valid_until, confidence, notes
   ▼
build.py  → static offline HTML folder (opens from file://, no server, no account):
   Brief (1 page) · Q01–Q10 with evidence links · Timeline · Decisions (proposal | decision | validation)
   · Contradictions (A vs B → resolution) · Actions (owner, due, evidence, commitment vs recommendation)
   · Sources browser · Uncertainties & limits · Versions: state_t0 (frozen) | state_t1 | diff
```
Optional, at the end: a natural-language question box using an LLM that is **only allowed to use `facts.yaml`**, must cite fact IDs, and answers "non documenté" otherwise. The challenge text mentions NL querying, but **the rubric does not score it**.

#### 7.1.8 The live event (10 pts): rehearse a drill, don't build a feature
The rubric wording (*"statut du problème, décision antérieure et nouvelle proposition… sans inventer d'approbation ni fermer d'autres conditions"*) suggests an event like "a re-test fails / a fix is delivered, and Boréal proposes a new date". Prepare `events/` + `diff.py` and rehearse 4–5 fake events:

| Fake event | Correct handling |
|---|---|
| Boréal delivers the ACC-303 fix in a new build | ACC-303: fix delivered, **not validated**. Condition 2 still open. Action: Mélissa re-tests. Nothing else changes. |
| SEC-210 re-test fails; Boréal proposes 29 Oct | **22 Oct remains the approved date** until the committee decides. 29 Oct is a *proposal*. Impacts: Q01, Q08, brief, risk R-02. Actions: convene committee, hold communications. |
| Runbook v2 received with rollback | Received ≠ approved (Olivier must approve). Step 5 status unknown. |
| Committee approves CR-04 | Authorised amount becomes 222k; INV-003's mobile line becomes payable **from the approval date only**. |

Each time output: `state_t1`, a diff table (changed facts / affected answers / affected actions / new or resolved contradictions), and the **unchanged baseline**.

#### 7.1.9 24-hour plan (3–4 people)
| Hours | Work |
|---|---|
| 0–3 | Script the extraction. **Two people answer Q01–Q10 independently**, then reconcile. Start `facts.yaml`. |
| 3–8 | Finish `facts.yaml` (timeline, decisions, actions, contradictions). `build.py` generates the static site. |
| 8–12 | Versioning + `diff.py`; rehearse the fake events. **Adversarial review:** one person tries to downgrade every answer from 5 to 3. |
| 12–16 | 1-page brief; action table; uncertainties page. |
| 16–19 | Optional NL Q&A over facts (citations mandatory). |
| 19–24 | User guide, zip export tested on a clean machine, demo rehearsal with a timed live-event drill, prepare answers about limits. |

#### 7.1.10 Do NOT build
Graph database (Neo4j), vector DB / RAG pipeline, fully automatic ingestion of every format (manual steps are allowed if documented), fancy UI, multi-project comparison (mentioned as a bonus but not scored), anything needing a paid account for the jury.

#### 7.1.11 Variants
- **Conservative:** curated `facts.yaml` + static HTML + answers + diff script. Expected 80–88.
- **Aggressive:** plus LLM-assisted extraction with human validation, NL Q&A with forced citations, auto-generated executive brief. Expected 85–92; the gain is mostly on "Use".
- **Differentiating idea:** an **"authority × time" resolver**. Each claim carries an authority level and an effective date, so every contradiction's resolution is computed and explained ("Plan v3 overridden by committee decision M04: higher authority, earlier decision, plan never updated"). Add a **"claimed but not yet proven" view**: fixes announced, fixes delivered, proposals. That is exactly what the jury tests.

#### 7.1.12 Verdict and honest critique
- Highest expected score, lowest risk, no uncertainty on data format (we've seen all of it).
- It does **not** use our ML/forecasting edge. The work is careful reading and clear information design.
- Many teams will get most facts right with an LLM. **Ranking will come down to nuance (3 vs 5 pts per answer) and the live event.** Our trap list is the edge.

---

### 7.2 JADCO / Collection Équinoxe

#### 7.2.1 The problem
Estimate the **2026 rent increase** for 6 rental buildings: Daniel-Johnson, Lévesque and Saint-Élzéar in Laval; Le Carlyle in Mont-Royal; Westpark in Pointe-Claire; The Met in Ottawa. The data comes from their Yardi CRM. The jury wants a **defended definition** of "rent increase" and a **method validated by backtest** (2023, 2024, 2025), not just a number. Python + Jupyter mandatory, starting from `starter.ipynb`.

#### 7.2.2 Official scoring [verified]
| Criterion | Points |
|---|---|
| Definition of the increase (named and defended) | 10 |
| Data analysis & mix effect (column understanding/renaming, charts, composition effect explained) | 15 |
| Same-unit matching on the right key (`sPropCode + sUnitCode`) | 15 |
| Concessions (`sRentEffective`, or justify using contract rent) | 10 |
| Renewals vs relocations (split + interpretation under the applicable rules) | 10 |
| **External data** (integrated and reconciled with the internal trend) | **15** |
| **Forecast & backtest** (justified method, stated assumptions, backtest on past years) | **15** |
| Notebook quality (structure, naming, **no magic numbers**, Markdown at each data-modifying step) | 10 |

Bonuses: treat Ottawa under Ontario rules; forecast by building and bedroom count; confidence interval or scenarios; small dashboard.
Penalties: magic numbers, bad naming, matching on `sSite + sUnitCode`, misuse of variables, missing Markdown.
Deliverables: notebook with `estimate_2026()` and `backtest()` completed; model file if any; README (how to run, library versions, public sources, AI tools); presentation; **final 2026 % with its definition**. **No CRM data in deliverables.**

#### 7.2.3 What's in the data [verified]
| File | Rows × cols | Content |
|---|---|---|
| `equinoxe_listings.csv` | 1,061 × 29 | One row per unit: beds, baths, sqft, floor, type, current asking rent |
| `equinoxe_lease_history.csv` | 4,302 × 35 | Leases 2017–2025: `sRent` (contract), `sRentEffective`, `sConcession`, sign/from/to dates, term, `sTermSeq`, `sRenewal` |
| `equinoxe_concessions.csv` | 3,560 × 18 | Concession lines: code, amount, date range, months |
| `equinoxe_asking_history.csv` | 3,857 × 14 | Monthly asking rent per unit |

- **Zero null values** anywhere. Clean, curated, pedagogical dataset.
- **9 property codes for 6 buildings:** `dj1`, `dj2` = Daniel-Johnson; `stelz1`, `stelz2`, `stelz3` = Saint-Élzéar; `carlyle`, `levesque`, `metcalfe` (= The Met, Ottawa), `wsp1` (= Westpark).
- `hUnit` is unique per (`sPropCode`, `sUnitCode`).

Leases per property code and start year:

| sPropCode | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|
| carlyle | | | | | | | 155 | 195 | 185 |
| dj1 | | | 48 | 57 | 57 | 55 | 57 | 58 | 59 |
| dj2 | | | | | | 65 | 77 | 78 | 71 |
| levesque | | 64 | 82 | 85 | 84 | 82 | 83 | 84 | 77 |
| metcalfe (Ottawa) | | | | | | | 104 | 128 | 120 |
| stelz1 | 93 | 108 | 103 | 104 | 101 | 101 | 99 | 96 | 36 |
| stelz2 | | | 95 | 127 | 119 | 119 | 118 | 119 | 118 |
| stelz3 | | | | | | | | | 134 |
| wsp1 | | | | | | | | 144 | 158 |

Three premium buildings enter in 2023–2024 (Carlyle, The Met, Westpark). That is the **mix effect**: the median rent jumps (1,912 → 2,120 $ from 2022 to 2023) without existing tenants paying much more.

#### 7.2.4 What the starter notebook already does
It loads the data, explains the Yardi `h` (handle, for joins) vs `s` (stored value, for display) convention, shows the naive median (fake +10.8% in 2023), shows the mix shift, demonstrates the `sSite + sUnitCode` collision (248 of 1,061 units), implements `same_unit_growth()`, compares contract vs effective, and splits renewals vs turnover.
→ **About 50 points are "floor" points that almost every team will get.** Differentiation must come from: **external data (15), forecast + backtest (15), Ontario treatment (bonus), notebook rigour (10).**

#### 7.2.5 Findings from our probe [verified]
**Same-unit annualised median growth** (consecutive leases of the same unit, gap 6–24 months), contract vs effective, in %:

| Year | Type | n pairs | Contract | Effective |
|---|---|---|---|---|
| 2018 | turnover | 41 | 1.03 | 1.03 |
| 2018 | renewal | 47 | 3.00 | 2.66 |
| 2019 | turnover | 67 | 1.21 | 1.32 |
| 2019 | renewal | 92 | 2.55 | 1.96 |
| 2020 | turnover | 131 | 2.05 | 2.32 |
| 2020 | renewal | 200 | 3.43 | 3.28 |
| 2021 | turnover | 135 | 3.52 | 2.36 |
| 2021 | renewal | 220 | 4.77 | 4.00 |
| 2022 | turnover | 126 | 4.22 | 3.40 |
| 2022 | renewal | 224 | 5.71 | 4.98 |
| 2023 | turnover | 176 | 1.85 | 1.70 |
| 2023 | renewal | 237 | 2.58 | 2.27 |
| 2024 | turnover | 276 | 4.86 | 2.75 |
| 2024 | renewal | 422 | 2.94 | 1.38 |
| 2025 | turnover | 306 | **8.99** | **5.76** |
| 2025 | renewal | 473 | **6.41** | **2.91** |

What this tells us:
1. **Contract and effective diverge from 2024** (concessions get bigger, not just more frequent), exactly as the brief hints.
2. **Renewals led until 2022; turnover leads in 2024–2025**, also as the starter notes.
3. **Quebec renewals track the TAL rate** (the official recommended increase): 2023 2.6% vs TAL 2.3%; 2025 6.4% vs TAL 5.9%; 2024 2.9% vs TAL 4.0% (timing lag). The TAL rate is published each January, so it is a **near-exogenous leading indicator for renewals**. *(TAL 2023/2024/2025 = 2.3/4.0/5.9% from memory; fetch the official 2026 figure from tal.gouv.qc.ca.)*
4. **The Met (Ottawa)**: renewals +3.0% (2024) and +6.3% (2025), turnover +4.9% and +9.0%. Renewals are well above the Ontario guideline (2.5%). This is consistent with Ontario's **exemption from the guideline for units first occupied after 15 Nov 2018**, since the building only starts leasing in 2023 **[hypothesis: verify the building's first-occupancy date]**. Explaining this is the "Ontario rules" bonus, and it avoids blindly applying a cap that may not apply.
5. **The `stelz3` trap:** `stelz3` starts in 2025 and reuses **58 unit codes** of `stelz1`, but with different `hUnit` and different sqft. Matching on site + unit code would create fake "same-unit" pairs. Correct key: `sPropCode + sUnitCode` (or `hUnit`).
6. **Concessions:**

| Code | Count | Median amount ($) | Median months |
|---|---|---|---|
| PromoPay | 1,494 | −1,356.60 | **1** |
| freepark | 1,006 | −150.50 | 12 |
| Freeothe | 570 | −153.49 | 12 |
| freelock | 175 | −139.35 | 12 |
| freerent | 166 | −129.30 | 12 |
| Indem | 69 | −149.26 | 12 |
| freeappl | 44 | −143.59 | 12 |
| Referenc | 36 | −148.26 | 12 |

`PromoPay` is described as monthly but is actually a **one-shot lump sum** (~one month of rent, 1 month duration). It must be **amortised over the lease term** to compute an effective monthly rent. The others are monthly amounts over ~12 months.

#### 7.2.6 Proposed method
1. **Definition (to defend):** headline = **same-unit effective rent growth**, weighted by the expected 2026 renewal/turnover mix, reported **by province**. Secondary = same-unit contract growth. Why: the owner budgets **collected** revenue (effective), while contract rent is what TAL/Ontario rules and asking rents refer to.
2. **Mix effect:** shift-share decomposition of the median/mean change into *within-unit price change* vs *composition* (new buildings, bedroom mix). One clear chart.
3. **Structural forecast** (few parameters, deliberately):
   - Renewals, Quebec: `g_ren = α + β · TAL_t` (TAL 2026 is known).
   - Renewals, Ontario (The Met): market-driven if exempt, else guideline.
   - Turnover: `g_turn = γ + δ · market_t` (CMHC same-sample rent change, or CPI rent, by metro area).
   - Effective: apply a concession-intensity trend (share of leases with concessions × amortised concession / rent).
   - Blend: `G = w_ren · g_ren + w_turn · g_turn`, weights = observed renewal share.
4. **Benchmarks in the backtest:** naive (last year), 3-year mean, structural, optional GBM at lease level (keep it only if it beats structural out-of-sample).
5. **Backtest:** expanding window. To predict year *t*, use leases up to *t−1* plus public info published by January of *t*. Report MAE and bias per model and segment.
6. **Uncertainty:** low/central/high scenarios from the backtest error spread plus a unit-level bootstrap.

**Critical point:** only ~8 yearly data points and **3 backtest years**. Any model with many macro variables **will overfit**. The defensible story is a low-parameter structural model anchored on regulatory rates and CMHC same-unit indices that beats naive baselines out-of-sample. Saying this explicitly is itself "method" points.

**External sources to integrate** (all listed in the brief): CMHC/SCHL Rental Market Survey (rents and vacancy by bedroom count, Montréal and Ottawa CMAs, including same-sample rent change); Statistics Canada table 18-10-0004-01 (CPI, rent component, Québec and Ontario); TAL annual calculation; Ontario rent increase guideline; optional Kaggle "25000 Canadian rental housing market June 2024".

#### 7.2.7 24-hour plan
| Hours | Work |
|---|---|
| 0–2 | Run and **read** the starter; column dictionary and renaming; a config cell with every constant (no magic numbers) |
| 2–6 | Mix decomposition; same-unit pairs by year × renewal × province × building; concession amortisation |
| 6–10 | Download external data (TAL, Ontario guideline, StatCan CPI rent, CMHC RMS); reconciliation charts (internal vs external) |
| 10–15 | `estimate_2026()` + `backtest()` with baselines |
| 15–18 | Scenarios; forecast per building / bedroom count; Ontario section |
| 18–22 | Markdown at every data-modifying step; README (versions, sources, AI tools cited) |
| 22–24 | Presentation |

#### 7.2.8 Do NOT build
Deep learning, heavy hyper-parameter search, web scraping, a big dashboard (at most a per-building table or chart), dozens of macro variables. **Never commit the CRM data** to any repo.

#### 7.2.9 Variants
- **Conservative:** starter + clear definition + TAL/guideline/CPI reconciliation + naive vs structural backtest. 70–80.
- **Aggressive:** plus a unit-level hierarchical model (building/bedroom effects), CMHC same-sample integration, scenarios, per-building forecast. 80–90.
- **Differentiating idea:** a **contribution waterfall** for 2026 (renewal regulatory pressure + turnover repricing − concession drag ± mix), plus showing the TAL-anchored renewal model would have forecast 2025 within X points.

#### 7.2.10 Verdict
Low risk, excellent skill fit, clean data. But a crowded field with a high floor; we would win on external data, backtest rigour and the Ontario insight. Main risks: subjective weighting by the jury, and clunky external-data portals (download and cache early).

---

### 7.3 IVADO / EquiAlgo

#### 7.3.1 The problem
A (fictional) Quebec institution grants scholarships with an ML model at 88% accuracy. An audit found **48.4%** of applicants from Montréal and Capitale-Nationale get a scholarship, versus **27.3%** in remote regions (Bas-Saint-Laurent, Côte-Nord, Gaspésie). The average cote R (academic score) is 28.0 vs 27.3, which explains only part of the 21-point gap. Mission: **diagnose the bias, fix it, and propose a production monitoring plan.** All data is synthetic.

#### 7.3.2 Official scoring [verified]
| Section | Points | Judged by |
|---|---|---|
| Diagnostic rigour | 25 | Jury |
| **Technical solution** | **35** | **Automatic grader vs a hidden reference** |
| Governance & ethics | 25 | Jury |
| Pitch & code quality | 15 | Jury |

The 35 automatic points:
- **Fairness, 20 pts:** proportion of the baseline model's **equal-opportunity gap** that we close. Baseline gap = **0.270**.
- **Utility, 15 pts:** agreement with the hidden reference, scaled from "random draw respecting the budget" to "perfect".
- **Both are 0 if the grant rate on the 4,000 candidates is outside 36–44%.**

Key rules: `decision_octroi` (the historical decision) **is not the target**; it records what the biased committee did. The reference was built **independently of the committee**. Removing `region_administrative` doesn't work (the parity gap goes from 0.188 to only 0.181, or 0.173 if postal code is also removed) because distance, work hours, income and postal code carry regional information (**proxies**).
Deliverables (GitHub repo): `predictions.csv` (4,000 rows, 0/1); `audit_rapport.ipynb`; `model_corrige.py/.ipynb` with a **Pareto front** chart over several constraint settings; `presentation.pdf` (5-min pitch).

#### 7.3.3 What's in the data [verified]
- `donnees_demandes.csv`: 10,000 historical applications with `decision_octroi` (40% granted overall).
- `candidats_evaluation.csv`: 4,000 applications to decide (Montréal 1,551; Capitale-Nationale 821; Gaspésie 633; Bas-Saint-Laurent 504; Côte-Nord 491).
- 9 features: cote R, program (5), region (5), postal code (3 chars), family income, work hours/week, home–campus distance, first-generation flag. No nulls.

Region profile (historical data):

| Region | n | Grant rate | Mean cote R | Median distance (km) | Mean work hours | First-gen share |
|---|---|---|---|---|---|---|
| Bas-Saint-Laurent | 1,293 | 26.9% | 27.30 | 202.0 | 12.95 | 44.8% |
| Capitale-Nationale | 1,999 | 48.4% | 27.98 | 16.7 | 8.95 | 25.8% |
| Côte-Nord | 1,178 | 26.3% | 27.33 | 207.7 | 13.12 | 42.8% |
| Gaspésie–Îles-de-la-Madeleine | 1,529 | 28.4% | 27.35 | 204.4 | 13.00 | 45.3% |
| Montréal | 4,001 | 48.3% | 27.99 | 17.1 | 8.98 | 26.8% |

Distance alone almost perfectly separates remote from central regions, which is why removing the region column fails.

#### 7.3.4 Findings from our probe [verified]
Logistic regression on the historical decision (standardised features, `remote` = 1 for the 3 remote regions):

| Feature | Std. coefficient |
|---|---|
| cote_r_equivalent | **+4.129** |
| heures_travail_semaine | +0.761 |
| log(revenu_familial) | +0.738 |
| distance (raw + log) | +0.086 / +0.080 |
| revenu_familial (raw) | +0.061 |
| programme dummies, first-gen | ≈ 0 |
| **remote** | **−1.082** |

- Cross-validated AUC: logistic **0.957**, gradient boosting 0.951. The committee is essentially a **linear score with a regional penalty**.
- **Counterfactual neutralisation:** train *with* the region variable (so the model attributes the penalty to region rather than to the proxies), then **predict as if everyone were from a central region**, and threshold at the 60th percentile (40% grant rate). Group grant rates become **41.4% (central) vs 38.0% (remote)**, versus historical 48.4% vs 27.3%. The remaining 3.4-point gap is consistent with the small real difference in cote R. *(In-sample on the 10,000 historical rows; a quick check, not a final evaluation.)*

#### 7.3.5 Why this matters
- "Drop the region column" fails because of the proxies (shown in the brief itself).
- `fairlearn.ThresholdOptimizer`, the tool the brief suggests and most teams will use, equalises true-positive rates **relative to the biased historical labels**, not relative to the hidden reference.
- If the hidden reference is "the committee's rule minus the regional penalty" (the most plausible way to build synthetic data "independently of the committee"), the counterfactual model reproduces it almost exactly. That means **high utility and near-full gap closure at the same time**.
- It has a literature basis: Pope & Sydnor (2011), *Implementing anti-discrimination policies in statistical profiling models*. Use the protected attribute when estimating, neutralise it when predicting.

#### 7.3.6 Plan
- **Diagnostic:** Oaxaca–Blinder-style decomposition of the 21-pt gap (explained by legitimate features vs unexplained penalty); proxy audit (predict region from each feature, AUC per feature); calibration by group.
- **Mitigation:** counterfactual model vs ThresholdOptimizer vs ExponentiatedGradient; sweep a blending parameter λ between historical and neutralised scores → **Pareto front**. Enforce a 40% grant rate.
- **Fairness metric choice:** equal opportunity. It is what the grader measures, and it is defensible ("among deserving candidates, the same chance in every region"). Explain why demographic parity is not chosen (groups differ slightly in cote R).
- **Governance:** monitoring per region (grant rate, score drift, appeals), human review of borderline cases, model card, documented limits (unknown reference, binary region grouping, intersectionality with first-gen/income).

#### 7.3.7 Risks, critique, variants
- The reference is hidden. The positive effects of income and work hours might themselves count as "bias" in the reference. We cannot know.
- 65% of points are jury-judged narrative, where teams look alike.
- **If the leaderboard F1 is computed against the reference** (§4 question 3), we can compare submissions (historical vs neutralised) to confirm the hypothesis.
- **Conservative:** ThresholdOptimizer + Pareto front (what most will do). **Aggressive:** counterfactual + λ-sweep + leaderboard probing. **Differentiating idea:** the counterfactual framing plus a decomposition showing how much of the gap is merit vs penalty.

---

### 7.4 CorroborAI (Loto-Québec)

#### 7.4.1 The problem
Compare employee records between **System A (HR master)** and **System B (time & scheduling)**. Apply the business rules that explain legitimate differences, and produce a report showing **only the real data errors to investigate**, each with the rule applied and a justification. AI should help with cases rules can't resolve.

#### 7.4.2 Official scoring [verified]
| Criterion | Points |
|---|---|
| Accuracy (compliant / justified gap / real anomaly) **on a test set** | 35 |
| Relevance of AI use (real value beyond static rules) | 25 |
| Explainability & traceability | 20 |
| Technical quality & usability | 20 |

Constraints: source files read-only; every verdict must be explained; business rules must be traceable; must distinguish **deterministic rule** vs **AI-assisted** verdicts; only fields in the mapping are compared; **no confidential/personal data sent to an unauthorised external service**. Demo must show at least one compliant case, one automatically justified gap and one real anomaly.

#### 7.4.3 What's in the data [verified]
| File | Content |
|---|---|
| `Employe_Source_Anonymise_VF.xlsx` | 23 rows × 33 cols (French column names). Several rows per employee: assignment type P (primary), A (temporary), S (secondary). |
| `Employe_Destination_Anonymise_VF.xlsx` | 22 rows × 79 cols (English names, ~48 `customAttribute_XX`); about 25 columns are actually mapped |
| `Mapping.xlsx` | 4 sheets: field mapping + rules; "situation d'emploi" status rules; join spec for position detail; join spec for status motifs |
| `détail_du_poste.xlsx` | Position history, **stored as CSV text inside a single Excel column**, dates as Excel serial numbers (e.g. 23604) |
| `Motif de la situation d'emploi.xlsx` | 92 rows: status code ↔ external code ↔ access-management code |
| `Présentation - CorroborIA LQ.pptx` | 1 slide, nothing new |

Examples of mapping rules:
- **Email** = first letter of first name + last name + last 3 digits of the code + `@loto-quebec.com`, accents removed.
- **contractTypeCode**: lookup table mapping each combination of `EstPermanent`, `EstTempsPlein` and employment category to a destination code.
- **isPrimaryAssignment / isTemporaryAssignment** from assignment type P / A / S.
- **assignmentStartDate** = the earlier of the position start date and the date the current administrative unit became effective (detected as a change in the position history; else the earliest effective date).
- **assignmentEndDate / termEndDate** = the earlier of the position end date and the day before the next history record, if that next record has a different administrative unit.
- **Status** ("Actif" / "Absence complète") and expected return date from the status table via the access-management code.

#### 7.4.4 Traps and risks
- Parsing traps: CSV-in-Excel, serial dates, mojibake (`Absence complÃ¨te` in the destination), accents to strip, several rows per employee, unmapped columns to ignore.
- **Ground-truth ambiguity:** some destination values (position names, emails) match neither the source codes nor the documented construction rule (e.g. emails carry a test-environment prefix). These may be **anonymisation artefacts** rather than planted anomalies, and we can't tell which. The 35 accuracy points are on a test set we don't see.
- **AI relevance (25 pts):** the rules are almost fully deterministic on ~23 records, so justifying AI honestly is hard.
- Personal data cannot go to external services, so we'd need a local LLM (setup risk).

#### 7.4.5 Plan, variants, verdict
- **MVP:** canonical schema → per-field comparison → rule engine (each rule has an ID and leaves a trace) → status {COMPLIANT, JUSTIFIED_GAP(rule), ANOMALY, AMBIGUOUS} → local LLM **only for AMBIGUOUS** (explanation + priority) → Excel report + small Streamlit viewer.
- **Differentiating idea:** an expert's correction becomes a candidate new rule (the brief's bonus), plus a priority/confidence score.
- **Verdict:** 55–70, medium risk. Solid engineering, but two scoring components (hidden test set, AI relevance) are partly out of our control.

---

### 7.5 DayOne / Offline Midwife

#### 7.5.1 The problem
Prototype a **WhatsApp-style agent** that photographs pages of a paper maternal registry, extracts structured data with a **status and confidence per field**, guides the midwife through verification (confirm / correct / retake / manual entry), works **offline-first** (encrypted local storage, queue, sync), and links visits of the same woman via a random code. No medical prediction (out of scope). No direct identifiers stored.

#### 7.5.2 Official scoring [verified]
| Criterion | Points |
|---|---|
| Extraction quality (per-field accuracy on a test set; handwriting & print; FR, AR, EN) | 30 |
| Uncertainty handling (correct statuses: CONNU, INCONNU, NON_FOURNI, ILLISIBLE, NON_APPLICABLE, À_RÉVISER; relevant confidence) | 20 |
| Conversational verification flow (confirm / correct / retake, follow-ups, manual entry, multi-page) | 20 |
| Offline robustness (nothing lost; queue, states, sync) | 15 |
| Patient linking & privacy | 10 |
| Code & docs | 5 |

Record lifecycle required: CAPTURÉ → EN_ATTENTE_IA → TRAITÉ_IA → À_RÉVISER → VALIDÉ → PATIENTE_LIÉE → ENREGISTRÉ → SYNCHRONISÉ, plus failure states.

#### 7.5.3 What's in the data [verified]
- 129 images (corrected 2026-10-03): **124 clean A4 renders** (`dossiers_specimen_10_patientes-NN*.png`, 80 unique pages = 10 fictional patients × 8 page types, 44 byte-identical duplicates) + **5 real phone photos** (`1-1…1-5.jpg`) of a pink paper booklet of a *different* registry model, handwritten in blue ink. The specimen PDF has a text layer with the filled values, i.e. exact per-page ground truth (see `dayone-participants/strat1.md`).
- `maternal_registry_synthetic.csv`: 200 rows × 31 numeric columns (age, education, gravidity, BMI, mean BP, hemoglobin, fasting glucose, HIV/syphilis/hep C results, birth weight…). It looks like a **risk-modelling table, not per-page ground truth for the images.** We could not measure our extraction accuracy without hand-labelling pages.
- Photos contain direct identifiers (CIN, address) that must be **redacted and never stored**.

#### 7.5.4 Verdict
Good product challenge with a clear rubric, but: extraction quality on handwriting is a black box until tested; 5 subsystems (extraction + schema, chat UI, offline state machine, encryption, patient linking, multi-page sessions); weak use of our quant edge. 45–65, high risk.
If chosen: schema and statuses first; a strong VLM with per-field confidence; the state machine in pure Python with tests; a Gradio chat mock; **skip** a real WhatsApp integration.

---

### 7.6 SN-SF / OptiFrame

#### 7.6.1 The problem
A **mobile web app** (public HTTPS URL + QR code, no install, Android and iOS) that photographs a recycled eyeglass lens next to a reference object, corrects perspective, segments the (transparent) lens, measures it to ~1 mm, and generates a **3D-printable frame (STL)** fitting two possibly different lenses. No data provided: we invent the capture rig, the dataset and the ground truth (calliper measurements).

#### 7.6.2 Official scoring [verified]
| Criterion | Points |
|---|---|
| **Measurement accuracy** (mean abs. error on A and B of the jury's 2 lenses: 30 pts if ≤1 mm, down to 0 at 4 mm) | **30** |
| Contour quality (SVG 1:1 printout, the lens must match) | 5 |
| Robustness (several shots, angles, lighting, a jury pair) | 10 |
| Data & AI (dataset ingenuity, model training, metrics, licences) | 15 |
| Mobile web app (QR, Android + iOS, clear UI, useful error messages) | 15 |
| Generated frame (closed mesh, rims match, bridge and hinges, printable) | 10 |
| Code quality | 5 |
| Presentation | 10 |

The jury opens the app **on their own phone**, reassembles our capture rig (<2 min), photographs the samples and compares to calliper values. Result in <30 s.

#### 7.6.3 Verdict
Many single points of failure (physical rig, browser camera on two OSes, HTTPS hosting, transparent-object segmentation, polygon offsetting, CAD/STL), none of them our core skills. 25–75 with very high variance. Choose only if someone has real OpenCV/ArUco **and** three.js/CAD experience.

---

### 7.7 L2C / plans vs shop drawings

#### 7.7.1 The problem
Compare reinforced-concrete **structural plans** with the subcontractor's **rebar shop drawings**. Extract rebar info (element, mark, diameter, quantity, spacing, length, sheet, X/Y position) into a **prescribed JSON schema**, match elements across the two, classify (compliant / non-compliant with detail / missing / added), and generate a **PDF report per plan sheet**.

#### 7.7.2 Official scoring [verified]
| Criterion | Points |
|---|---|
| Detection of non-conformities (recall weighted more than precision) on the **unseen evaluation project** | 30 |
| Extraction & JSON quality (completeness, attributes, X/Y, schema) | 20 |
| Report usefulness | 15 |
| Technical quality & code | 15 |
| Robustness & generalisation (unseen project, all 5 element types) | 10 |
| Presentation (10-min live run on the evaluation project) | 10 |

Hard constraints: Python core; **documents must not be uploaded to cloud services or external AI APIs**; no commercial-licence software; **data deleted from our machines at the end**.

#### 7.7.3 What's in the data [verified]
| Project | Plan pages | Page size (pt) | Words in plan text layer | Shop-drawing PDFs |
|---|---|---|---|---|
| CLP | 28 | 3455 × 2591 | 32,648 | 12 |
| EspCa3B | 72 | 1728 × 2592 | 44,504 | 54 |
| LIGREP | 40 | 3455 × 2591 | 38,177 | 42 |
| WP2 | 50 | 3456 × 2592 | 50,400 | 29 |

- **Decisive fact: 119 of the 137 shop-drawing PDFs have fewer than 20 extractable words and 0 embedded images.** The text is drawn as **vector curves**. Reading them needs high-DPI rasterisation plus **local** OCR (cloud AI forbidden) on sheets up to 48"×36".
- `CLP_dismatch.xlsx` gives 6 example mismatches (sheets S-050 to S-603: wrong bar count, wrong diameter, wrong spacing). The notation is heterogeneous: count × diameter, spacing in inches, layer prefixes, parenthesised counts. *(Rows not reproduced here: L2C documents are confidential.)*

#### 7.7.4 Verdict
15–45, extreme risk. A single unexpected PDF problem could eat the whole 24 hours. **Avoid.**

---

### 7.8 Propolys / Security & AI pitch

- **Task:** invent a startup using AI for a security problem (cyber, physical, infrastructure, fraud, deepfakes, crisis response, protecting AI systems…). Deliver a **3-minute pitch + max 3 slides** by **16:00**. Prize $400.
- **Criteria:** clear and original idea, link with security, entrepreneurial potential, pitch quality. No technical deliverable.
- **Verdict:** low technical risk, fully subjective, doesn't use our edge. Worth it **only as a side quest** (one person, ~3 h) if multi-challenge entry is allowed. Pick a precise, measurable problem with a clear buyer and ROI. Examples: vendor/invoice-fraud detection for municipal procurement; deepfake-voice "CEO fraud" protection for SMB finance teams.

---

## 8. Comparison matrix

Scale 1 (worst) to 5 (best). For risk, work and data dependency, **5 = lowest** (best).

| Criterion | NOVA | JADCO | EquiAlgo | CorroborAI | DayOne | OptiFrame | L2C | Propolys |
|---|---|---|---|---|---|---|---|---|
| Expected score | **5** | 4 | 4 | 3 | 2 | 2 | 1 | ? |
| Execution risk (5 = low) | **5** | **5** | 4 | 3 | 2 | 1 | 1 | 5 |
| Technical differentiation | 3 | 4 | **5** | 3 | 3 | 4 | 4 | 1 |
| Demo quality | 4 | 3 | 3 | 3 | 4 | **5** | 3 | 3 |
| Robustness of the result | **5** | **5** | 4 | 3 | 2 | 1 | 1 | 4 |
| Amount of work (5 = low) | 4 | 3 | 4 | 3 | 2 | 1 | 1 | 5 |
| Dependency on uncertain data (5 = low) | **5** | **5** | **5** | 3 | 2 | 2 | 1 | 5 |
| Use of our skills | 2 | **5** | **5** | 3 | 2 | 1 | 2 | 1 |
| How many average teams will produce something similar | High | High | High | Medium | Medium | Low | Low | High |

---

## 9. Final recommendation and decision tree

```
Can we enter more than one challenge?
├── YES → NOVA (2 people, ~25–35 person-hours) + JADCO (2 people)
│         (+ Propolys pitch as a 3-hour side quest if allowed)
└── NO
    ├── Prizes per challenge AND NOVA looks crowded?
    │   ├── YES → EquiAlgo (objectively scored edge) or JADCO (skills edge)
    │   └── NO  → NOVA
    └── Team prefers to use forecasting skills / learn? → JADCO
```

1. **NOVA** maximises expected score with the lowest risk. The 50 factual points are essentially pre-secured (§7.1.4); the other 50 are an explicit checklist. Spend the saved time on nuance, the live-event drill and the adversarial review.
2. **JADCO** if we want our forecasting/backtesting skills to be the differentiator, accepting ~50 floor points shared with everyone.
3. **EquiAlgo** if prizes are per challenge and the others look crowded: it has the only *objectively scored* lever we can exploit (§7.3.4–7.3.5).
4. **Avoid** L2C, OptiFrame and DayOne unless someone has specific prior experience.

*AI tools: this analysis was produced with Claude Code (Anthropic). NOVA, JADCO and OptiFrame require citing AI tools; cite it if you reuse this work.*

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **Barème / rubric** | The official scoring grid of a challenge. |
| **NOVA / Boréal** | The fictional project (a request-tracking portal) and its fictional vendor, in the Loto-Québec "Projet 360" challenge. |
| **ADR** | Architecture Decision Record: a document recording an architecture decision (ADR-007 = data location). |
| **CR (demande de changement)** | Change request. CR-01 approved (24k), CR-04 draft (18k). |
| **Go-live / mise en production** | Deployment to production. |
| **Runbook / rollback** | Step-by-step deployment procedure / procedure to revert a deployment. |
| **Baseline (NOVA)** | The project state as of 30 Sept 09:00, to be kept intact after the new event. |
| **Yardi** | Property-management CRM used by JADCO. `h*` columns = handles (IDs for joins), `s*` = stored values (for display). |
| **Same-unit growth** | Rent change of a unit compared with its own previous lease; removes the mix effect. |
| **Mix / composition effect** | An average or median moves because the set of units changes (e.g. premium buildings added), not because rents rose. |
| **Contract rent (`sRent`) vs effective rent (`sRentEffective`)** | Rent on the lease vs rent actually collected after concessions. |
| **Concession** | Discount: free month, free parking, free locker… `PromoPay` = one-shot lump sum. |
| **Renewal vs turnover (relocation)** | Same tenant renewing (`sRenewal = 1`) vs a new tenant after a vacancy (`sRenewal = 0`). |
| **TAL** | Tribunal administratif du logement (Quebec rental board). Publishes the recommended yearly increase for sitting tenants. |
| **Ontario rent increase guideline** | Ontario's yearly cap for sitting tenants. Does not apply to units first occupied after 15 Nov 2018. |
| **CMHC / SCHL** | Canada Mortgage and Housing Corporation. Rental Market Survey: rents and vacancy by metro area and bedroom count. |
| **CPI rent (StatCan 18-10-0004-01)** | Consumer price index, rent component, by province. |
| **Backtest (expanding window)** | Pretend it is year *t*, use only data available then, forecast *t*, compare with reality; repeat for several years. |
| **MAE / bias** | Mean absolute error / average signed error of forecasts. |
| **Equal opportunity (EO) gap** | Difference between groups in the true-positive rate: among deserving candidates, the share who get the scholarship. |
| **Demographic parity** | Same grant rate in every group, regardless of merit. |
| **Proxy variable** | A feature that indirectly encodes the sensitive attribute (distance ≈ region). |
| **ThresholdOptimizer / ExponentiatedGradient** | fairlearn tools: per-group thresholds after training / retraining under a fairness constraint. |
| **Counterfactual neutralisation** | Train with the sensitive attribute, then predict with it set to the same value for everyone. |
| **Pareto front** | Set of best trade-offs between two objectives (here fairness vs utility) obtained by sweeping a parameter. |
| **Oaxaca–Blinder decomposition** | Splits a gap between groups into the part explained by characteristics and an unexplained part. |
| **AUC** | Area under the ROC curve: how well a score separates positives from negatives (1 = perfect, 0.5 = random). |
| **VLM / OCR** | Vision-language model / optical character recognition. |
| **ArUco** | Printable square markers that OpenCV detects precisely; used for scale and perspective. |
| **STL** | 3D mesh file format for 3D printing. |
| **Text layer (PDF)** | Real text characters inside a PDF. Without one, the text is just drawn shapes and needs OCR. |

---

## Appendix A: reproduce the probes

Run from the project root `codeml/`. Python 3 with pandas, numpy, scikit-learn, openpyxl; `pdftotext`/`pdfinfo` (poppler-utils).

### A.1 Extract all instruction PDFs to text
```bash
mkdir -p /tmp/consignes_txt
for f in $(find . -name "consignes*.pdf"); do
  n=$(echo "$f" | sed 's|^\./||; s|/|__|g; s|\.pdf$||')
  pdftotext -layout "$f" "/tmp/consignes_txt/$n.txt"
done
```

### A.2 NOVA: unzip and read everything
```bash
mkdir -p /tmp/nova && unzip -oq loto-quebec-nova-participants/NOVA_ETUDIANTS.zip -d /tmp/nova
cat /tmp/nova/Projet360_NOVA_ETUDIANTS/README.txt     # rubric + the 10 questions
```
```python
# Emails (with attachment names), from inside /tmp/nova/Projet360_NOVA_ETUDIANTS
import email, glob
from email import policy
for f in sorted(glob.glob('01_Courriels/*.eml')):
    m = email.message_from_bytes(open(f, 'rb').read(), policy=policy.default)
    print('###', f, '|', m['From'], '|', m['Date'], '|', m['Subject'])
    for part in m.walk():
        if part.is_multipart():
            continue
        if part.get_filename():
            print('  [ATTACHMENT]', part.get_filename())
        elif part.get_content_type().startswith('text/'):
            print(part.get_content().strip())

# Spreadsheets including cell comments
import openpyxl
for f in sorted(glob.glob('*/*.xlsx')):
    for ws in openpyxl.load_workbook(f, data_only=True).worksheets:
        print('###', f, ws.title)
        for row in ws.iter_rows():
            print([(c.coordinate, c.value) for c in row if c.value is not None])
            for c in row:
                if c.comment:
                    print('  COMMENT', c.coordinate, c.comment.text)
```
PDFs: `pdftotext -layout file.pdf -`. Look at the PNGs in `03_Tickets/` (especially `OPS-601_runbook.png`).

### A.3 JADCO: same-unit growth by year × renewal (and by province)
```python
import pandas as pd
L = pd.read_csv('jadco-participants/equinoxe_lease_history.csv',
                parse_dates=['sSignDate', 'sLeaseFrom', 'sLeaseTo'], dtype={'sUnitCode': str})
L['yr'] = L.sLeaseFrom.dt.year
L = L.sort_values(['sPropCode', 'sUnitCode', 'sLeaseFrom'])
g = L.groupby(['sPropCode', 'sUnitCode'])              # correct key, never sSite + sUnitCode
L['prev_rent'] = g.sRent.shift()
L['prev_eff'] = g.sRentEffective.shift()
L['prev_from'] = g.sLeaseFrom.shift()
m = L.dropna(subset=['prev_rent']).copy()
m['gap_m'] = (m.sLeaseFrom - m.prev_from).dt.days / 30.44
m['g_contract'] = (m.sRent / m.prev_rent) ** (12 / m.gap_m) - 1
m['g_effective'] = (m.sRentEffective / m.prev_eff) ** (12 / m.gap_m) - 1
ok = m[(m.gap_m > 6) & (m.gap_m < 24)]
print(ok.groupby(['yr', 'sRenewal']).agg(n=('g_contract', 'size'),
      g_contract=('g_contract', 'median'), g_effective=('g_effective', 'median')).round(4))
print(ok.groupby(['sState', 'yr', 'sRenewal']).g_contract.median().unstack().round(4))

# The stelz1 / stelz3 trap
s1 = set(L[L.sPropCode == 'stelz1'].sUnitCode); s3 = set(L[L.sPropCode == 'stelz3'].sUnitCode)
print('unit codes shared by stelz1 and stelz3:', len(s1 & s3))
```

### A.4 EquiAlgo: committee model and counterfactual
```python
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
d = pd.read_csv('equialgo-participants/data/donnees_demandes.csv')
remote = ['Bas-Saint-Laurent', 'Cote-Nord', 'Gaspesie-Iles-de-la-Madeleine']
d['remote'] = d.region_administrative.isin(remote).astype(int)
cols = ['cote_r_equivalent', 'programme_etudes', 'revenu_familial_estime', 'heures_travail_semaine',
        'distance_domicile_campus_km', 'premiere_generation_universitaire', 'remote']
X = pd.get_dummies(d[cols], columns=['programme_etudes'], drop_first=True).astype(float)
X['log_rev'] = np.log(X.revenu_familial_estime); X['log_dist'] = np.log1p(X.distance_domicile_campus_km)
mu, sd = X.mean(), X.std(); Xs = (X - mu) / sd
y = d.decision_octroi
lr = LogisticRegression(max_iter=2000).fit(Xs, y)
print(pd.Series(lr.coef_[0], Xs.columns).round(3).sort_values())
print('CV AUC', cross_val_score(LogisticRegression(max_iter=2000), Xs, y, cv=5, scoring='roc_auc').mean())
# Counterfactual: everyone treated as non-remote, 40% grant rate
Xcf = X.copy(); Xcf['remote'] = 0
p = lr.predict_proba((Xcf - mu) / sd)[:, 1]
pred = (p >= np.quantile(p, 0.60)).astype(int)
print(d.assign(pred=pred).groupby('remote').pred.mean(), d.groupby('remote').decision_octroi.mean())
```

### A.5 L2C: count extractable text in shop drawings (takes a few minutes)
```bash
find l2c-participants -path '*/DA/*' -name '*.pdf' -print0 |
  while IFS= read -r -d '' f; do pdftotext "$f" - 2>/dev/null | wc -w; done |
  awk '{n++; if ($1 < 20) z++} END {print n" shop drawings, "z" with <20 words of text"}'
```

### A.6 CorroborAI: read the workbooks without pandas
```python
import openpyxl, glob
for f in sorted(glob.glob('corroborai-participants/*.xlsx')):
    for ws in openpyxl.load_workbook(f, read_only=True, data_only=True).worksheets:
        rows = list(ws.iter_rows(values_only=True))
        print('###', f, ws.title, len(rows), 'rows')
        for r in rows[:10]:
            print([c for c in r if c is not None])
# Note: détail_du_poste.xlsx has CSV text in column A; split on ',' and convert serial dates
# with pd.to_datetime(serial, unit='D', origin='1899-12-30').
```

---

## Appendix B: file inventory per folder

| Folder | Size | Files |
|---|---|---|
| `corroborai-participants/` | 380 KB | 5 xlsx (source, destination, mapping, status motifs, position detail), 1 pptx, `consignes.pdf` |
| `dayone-participants/` | 15 MB | `consignes-fr-en.pdf`, `data/maternal_registry_synthetic.csv`, `data/Paper Registry/` (129 images + 1 PDF) |
| `equialgo-participants/` | 1.2 MB | README/LISEZMOI, `baseline_model.ipynb`, `requirements.txt`, consignes FR/EN, `data/donnees_demandes.csv`, `data/candidats_evaluation.csv` |
| `jadco-participants/` | 5.1 MB | 4 CSVs, `starter.ipynb`, `consignes.pdf`, `presentation-jadco.pdf` |
| `l2c-participants/` | 288 MB | consignes FR/EN, 4 projects (CLP, EspCa3B, LIGREP, WP2), each with a plan PDF and a `DA/` folder of shop drawings by element type; `CLP/CLP_dismatch.xlsx` |
| `loto-quebec-nova-participants/` | 3.3 MB | `consignes.pdf`, `Image-defi-2.png` (illustration), `NOVA_ETUDIANTS.zip` (64 files + README + MANIFEST) |
| `optiframe-participants/` | 208 KB | `consignes.pdf` only |
| `propolys-participants/` | 296 KB | `consignes-fr-en.pdf` only |

Each folder also has a `manifest.json` (file list with sizes and SHA-256 hashes).

---

## Appendix C: confidentiality rules per challenge

| Challenge | Rule (from the instructions) |
|---|---|
| **JADCO** | The CRM subset **must not be shared, published or pushed to a public repo** (GitHub, Kaggle…), and must not be in the deliverables. Source files stay untouched; all transformations happen in the notebook. |
| **L2C** | Documents are confidential; **no upload to cloud services or external AI APIs**; **delete the data from our machines at the end of the event**. |
| **CorroborAI** | No confidential or personal data sent to an unauthorised external service. Source files are read-only. |
| **DayOne** | No real patient data to third parties; direct identifiers visible on paper (name, spouse's name, national ID, phone, address) must be ignored or redacted, never stored; local storage encrypted; original images and reference files not modified. |
| **EquiAlgo** | Synthetic data; the deliverable is a GitHub repo (public or shared with the judges). |
| **NOVA** | Fictional data; external AI tools allowed but must be declared; facts must come from the corpus. |
| **OptiFrame** | No personal data (faces, names, prescriptions) in our datasets; every dataset and model cited with its licence. |

**Never add the `jadco-participants/`, `l2c-participants/` or `corroborai-participants/` folders to this repo. They are git-ignored on purpose.**
