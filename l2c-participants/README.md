# l2c-rebar: from plans to shop drawings

Checks rebar shop drawings (dessins d'atelier) against the structural plans of a reinforced-concrete
building and tells the engineer where they disagree.

For one project it produces:

| Output | File | What it is |
|---|---|---|
| JSON database | `<project>_elements.json` | Every rebar annotation found on the plans and the shop drawings, in the Appendix A schema (sheet, page, x, y, element, bars) |
| Comparison | `<project>_comparaison.json` | Each element classified: compliant, non-compliant (with the discrepancy), missing from the shop drawings, added in the shop drawings |
| PDF report | `<project>_rapport.pdf` | Counts per plan sheet, one list of every finding by sheet and grid place (`axes B-12`), then each non-conformity with its location and an image extract of both drawings |
| Findings for a spreadsheet | `<project>_ecarts.csv` | One row per finding (semicolons, opens in Excel): sheet, grid place, element, discrepancy, severity, confidence, page and x, y on both drawings |
| Annotated PDFs | `annotes/*_annote.pdf` | Copies of the plan and shop drawings with each finding circled and linked to its counterpart |
| Details | `<project>_details.json` | What the schema has no room for: confidence, raw text, bounding boxes, storey |

Everything runs on the local machine. No document, text or image is sent anywhere.

## Quick start

```
python -m venv .venv
.venv\Scripts\activate            # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

python -m l2c_rebar run path\to\PROJECT                  # writes outputs\PROJECT\
python -m l2c_rebar evaluate path\to\PROJECT known.xlsx  # score against documented non-conformities
python -m l2c_rebar ui                                   # interface, with validation of uncertain findings
python -m l2c_rebar diff old.pdf new.pdf                 # what changed between two revisions of a shop drawing
pytest                                                   # tests (add -m "not slow" to skip the OCR test)
```

The project folder is expected as handed out for the challenge:

```
PROJECT/
  L2C_PLAN_STR_PROJECT.pdf          the structural plans (one PDF, one sheet per page)
  DA/
    Colonnes/*.pdf                  shop drawings, grouped by element type
    Dalles/*.pdf
    Fondations/*.pdf  ...
```

Useful options of `run`: `--ocr off` (skip pages without text, a few seconds per project), `--no-crops`
(report without image extracts), `--fast` (live demonstration: OCR at 200 dpi and no image extracts, about 25%
faster), `--config file.json` (override any field of `Config`, see `src/l2c_rebar/config.py`).

While OCR runs, the console shows the pages done and an estimate of the time left. A page the OCR cannot read
is left out (and retried on the next run) instead of stopping the analysis.

To try the tool without any confidential data, `notebooks/demo.ipynb` builds a made-up project with planted
errors and runs the whole pipeline on it.

### Live demonstration on an unseen project

1. Beforehand, with a network connection: `pytest -m slow` once, so the OCR models are downloaded and cached.
2. As soon as the project is handed over: `python -m l2c_rebar run path\to\EVAL --fast`. Pages with a text layer
   take seconds; pages without text take 10 to 25 s each on a DirectX 12 GPU, and the console shows the time left.
3. While OCR runs, walk through a development project's report: the list of findings by sheet and axes on the
   second page, one finding card with both image extracts, the annotated PDFs and their cross links.
4. When the run ends: `outputs\EVAL\EVAL_rapport.pdf`, then `python -m l2c_rebar ui` to validate uncertain findings.
5. Given a list of known non-conformities (sheet, grid location, plan value, shop value):
   `python -m l2c_rebar evaluate path\to\EVAL known.xlsx` prints, row by row, how far the tool got.

The slides (`python tools/build_slides.py`, PDF in `outputs/presentation/`) read their figures from `outputs/`.

### Reading sheets that have no text

Many shop drawings are CAD plots without a text layer. They are read by a local OCR model, in three steps
that each raised the share of callouts read exactly on a benchmark page (a sheet that has both drawn text and a
text layer, used as ground truth):

| Setting | Callouts read exactly |
|---|---|
| PP-OCRv4, 200 dpi (models bundled in the wheel; automatic fallback) | 57% |
| PP-OCRv5 with the English recogniser, 200 dpi | 79% |
| Same at 300 dpi (default) | 85% |

On vector plots the page is first redrawn with only its character-sized paths, so that dimension lines and bars
no longer cross the lettering (`pdf/vector.py`). Text that AutoCAD stores beside stroke lettering as
"AutoCAD SHX Text" comments is read directly when a plot kept it.

The OCR detector cuts a line of lettering at wide spaces: a shop callout `24 10M 10A12` and, further right,
`23@300` (23 spaces at 300 mm) come out as two boxes, and the tie spacing would be lost. Pieces of the same height
on the same row, closer than 1.5 text heights, can be joined back into one line (`pdf/ocr.py:join_row_fragments`,
`ocr_join_gap`); the grid is then read again, since two-digit bubbles also come out in pieces. Whether to join is
decided per element type, by agreement (see below): on the development projects it helps slabs and walls
everywhere, and column sheets on one project out of three.

The PP-OCRv5 models are fetched once by `rapidocr` on first use and then cached; run once with a network
connection before a demonstration. With a DirectX 12 GPU, a sheet takes 15 to 25 seconds instead of minutes:

```
pip uninstall -y onnxruntime
pip install onnxruntime-directml
```

OCR results are cached in `outputs/<project>/.ocr-cache/`, so a second run takes seconds.

## How it works

```
PDFs ──> text lines with coordinates ──> rebar callouts ──> elements ──> matching ──> comparison ──> report
         (text layer, else local OCR)    (grammar)          (layout,     (name or grid
                                                             grid)       crossing, else content)
```

| Step | Module | What it does |
|---|---|---|
| Read | `pdf/text.py`, `pdf/ocr.py`, `pdf/vector.py` | Text lines in displayed page coordinates (page rotation applied). A page with fewer than 25 words is read by OCR, tile by tile |
| Identify the sheet | `pdf/sheets.py`, `parsing/elements.py` | Sheet number from the PDF bookmark or the title block. Element type from the series (S-100 foundation, S-300 beam, S-400 shear wall, S-500 column, S-600 slab). For shop drawings, type and storey come from folder and file names |
| Parse callouts | `parsing/rebar.py` | Quantity, size, spacing, length, mark. Plans: `8-25M`, `12(6)-15M`, `10M@300 c/c`, `10M@12" c/c`. Shop drawings: `24 15M 15A12 @12"`, `2x4 25M 25A301`, `24 15M 20-6`. Spacings and lengths are converted to millimetres |
| Locate on the grid | `extract/grid.py` | On a column plan a column has no name: it is a filled rectangle at a grid crossing with a tag beside it. The grid is rebuilt from its bubbles, each tag is paired with its rectangle, and the column is named by its crossing (`B-3`), which is how shop drawings refer to it (`B-3`, `B/3`, `B-12, B-13`) |
| Group into elements | `extract/layout.py`, `extract/document.py` | Elsewhere, a callout belongs to the label above it in a schedule, on its row in a table, or next to it in a detail. Names written as a list share one detail. Callouts without a name become elements named by sheet zone |
| Match | `compare.py` | By element name or grid crossing when plan and shop drawings share them. Otherwise by content (size, quantity, spacing) within the same element type and storey |
| Compare | `compare.py` | Only attributes present on both sides. A plan quantity may be split over several shop marks; a plan in millimetres and a shop drawing in inches agree within 2% |
| Report | `report/` | ReportLab PDF, image extracts, annotated PDFs |
| Score | `evaluate.py` | Against a spreadsheet of documented non-conformities (sheet, grid location, plan value, shop value): recall, and for each miss the stage where it was lost |

### Classification and severity

Each element gets one of the four statuses required by the rules. Each discrepancy has a severity:

| Severity | Meaning |
|---|---|
| critique | Less steel than designed: fewer bars, smaller size, wider spacing |
| majeur | Bars not found, larger size substituted, shorter length |
| mineur | More steel than designed |

Each result carries a confidence between 0 and 1. Below 0.60 it is flagged "à valider": low OCR score, text read
by OCR combined with a match by content, uncertain storey, shared detail. Findings made by name also inherit how
well their element type reads overall: when fewer than 90% of the bars paired by name are confirmed for a type,
its findings lose confidence in proportion (the report states the agreement per type). The interface lets the
engineer confirm or reject the flagged results, and regenerates the report.

### Finding an element on the sheet

Every finding gives the plan sheet, the page, x, y in PDF points, a sheet zone (A1 to H6) and, on plan views, the
place on the building grid as the engineer reads it: the crossing (`axes B-12`) when the element stands on both
lines, otherwise the bay it falls in (`axes B-C/11-12`). The same place is in the comparison JSON (`axes_plan`,
`axes_atelier`) and in the details file (`axes`).

### Calibrated by agreement

Detailers have their own habits, and a new project brings new ones. Instead of fixed assumptions, the tool
relies on one fact: most of a shop drawing agrees with its plan. Where a convention is open, it tries the
readings and keeps the one that agrees most often.

1. **Names are used only when both sides share them.** If plan and shop drawings have no element name or grid
   crossing in common for a type, that type is compared by content instead.
2. **Storeys.** A shop sheet named `NIV 2@3` may detail the plan storey 2 or 3, and a building may skip a floor
   number. Shop storey names are mapped to plan storeys, in order, so as to agree as often as possible.
3. **Schedule or details.** A sheet of named elements is read both as a schedule (names in a header, bars under
   them) and as rows of details (each titled close by); the better reading is kept, per element type.
4. **Per element or totals.** A detail drawn once for several elements (`B-12, B-13`) may list the bars of one
   or the total for all; the reading that agrees most is kept. A shop quantity that is an exact multiple of the
   plan quantity is taken as such a shared detail, not as a surplus.
5. **Mass disagreement means a wrong pairing.** If fewer than half of the bars of the pairs made by name are
   confirmed for a type, the pairing is abandoned for that type and stated in the report. Counted per bar, not
   per element: a beam with eight bars and one misread disagrees as a whole, yet its pairing is plainly right.
6. **Pieces of OCR lines.** Joining the pieces of a row that the OCR detector boxed separately gives callouts
   their spacing or quantity back, but on sheets laid out as tables it can glue neighbouring cells. Both readings
   are compared per element type, and the one that confirms more plan bars is kept (stated in the report).
7. **Missing elements are reported only where the sheet was read well.** When fewer than half of a sheet's
   elements are found in the shop drawings, the report says so once instead of listing each element.

## The JSON database

`<project>_elements.json` is a list of records with exactly the keys of Appendix A, validated by pydantic
(`models.Record`). `x`, `y` are PDF points from the top-left corner of the displayed page, at the centre of the
annotation. `source` is `plan` or `atelier`.

```json
{
  "id": "S-501_B-2@RDC_plan", "source": "plan", "fichier": "L2C_PLAN_STR_GRILLE.pdf", "feuillet": "S-501",
  "page": 1, "x": 416.9, "y": 372.9, "type_element": "colonne", "element": "B-2",
  "armature": [
    {"repere": null, "diametre": "25M", "quantite": 12, "espacement_mm": null, "longueur_mm": null},
    {"repere": null, "diametre": "10M", "quantite": null, "espacement_mm": 300, "longueur_mm": null}
  ]
}
```

For shop drawings, `feuillet` is the file name (plus the page for multi-page files), since those sheets carry no
plan sheet number.

## Assumptions

- Bar sizes are Canadian metric (10M to 55M). Spacing may be in millimetres or inches; both are stored in mm.
- Sheet series follow the rules: S-1xx foundations, S-3xx beams, S-4xx shear walls, S-5xx columns, S-6xx slabs.
  Other sheets are typed by their title, or extracted as `autre` and not compared.
- Shop drawing folders or file names contain the element type (Colonnes, Dalles, Fondations, Poutres, Murs,
  Refends, Semelles, Radiers) and, where relevant, the storey (`NIV 2`, `RDC`, `SS1`, `2@3`).
- Grid bubbles of a plan view are the largest short labels on the sheet, letters in one direction and numbers
  in the other.
- In `12(6)-15M` the number in brackets is a subset of the quantity, not a length.
- In a shop callout `24 15M 20-6`, `20-6` is a length in feet and inches; a three-digit number after the size
  is a bar mark.
- A shop drawing details more than the plan shows. A shop callout without a name and without a plan twin is
  counted, not reported as an addition.

## Known limitations

These are stated plainly because they decide how far the output can be trusted.

- **Slabs, foundations and beams are confirmed, not contradicted, by position.** Every callout of a plan view is
  placed on the building grid, and a plan callout whose twin is drawn within half a bay on the shop drawing is
  reported compliant. A different number at the same place is not reported: measured on a development project,
  only one slab callout in five has its twin that close (a detailer writes a bar where it starts, an engineer
  where it is needed), so it would mostly raise false alarms. What position cannot confirm falls back to
  matching by content within the storey, where a wrong quantity goes unnoticed if another callout carries it.
  On the one development project that came with a list of documented non-conformities, the tool finds the two
  column cases (right location, right values) and misses the slab, foundation and wall cases for this reason.
- **Slab callouts written without a bar size** (a bare `12(6)`) are read into the JSON and confirmed by position
  when possible; otherwise they are counted as not verified. A plain number beside a bar is not read yet.
- **The order of reinforcement layers is not compared** (plan "layer 1" against the shop drawing's layering).
- **Columns on intermediate grid lines.** A shop drawing may name a column `B.1-12` where the plan prints no `B.1`
  bubble. Such a column is named by its nearest labelled line on the plan and the shop element is reported as
  added. Naming plan columns by the bay they stand in (`grid_between`) is available but off: on the development
  projects position alone does not tell these columns from columns drawn off-centre.
- **Footings without a tag.** A footing linked to its table row or detail only by its drawn shape is not read; its
  bars are compared by content, not at its place.
- **OCR is imperfect.** Even at 85% of callouts read exactly, a dense sheet carries misreads. Every result that
  rests on OCR text says so and has its confidence lowered. On one development project the column sheets read by
  OCR disagree with the plan for a third of the columns, which is not credible: those findings need the engineer.
- **Layout rules are heuristics.** A callout is given to the nearest plausible label; dense sheets can mislead it.
- **Development was done almost entirely without reading the drawings.** The confidentiality rule forbids
  sending documents to external AI services, and this tool was written with an AI coding assistant. The
  assistant did not open the drawings or the list of known non-conformities: it worked from counts, from masked
  notation patterns (every digit replaced by 9, every letter by A), from a team member's description of a
  column tag (including one tag quoted as an example), and from the score returned by `evaluate`. One exception:
  a team member sent the assistant a screenshot of part of a slab plan. Nothing from it was copied into the
  code, the tests or this repository; it only confirmed that slab callouts are written along their bars.
- No model was trained. The OCR models are pre-trained, open-source ones (PP-OCR, Apache 2.0).

## Bonus features

- **Annotated PDFs**: findings circled on both documents, with a link from each tag to its counterpart.
- **Confidence score and validation**: `python -m l2c_rebar ui`; verdicts are saved in `validations.json`.
- **Revision comparison**: `python -m l2c_rebar diff old.pdf new.pdf --out changes.json`.

## Repository layout

```
src/l2c_rebar/
  cli.py  pipeline.py  config.py  models.py  compare.py  evaluate.py  validation.py  revisions.py
  synthetic.py  app.py
  pdf/        text.py  ocr.py  vector.py  sheets.py
  parsing/    rebar.py  elements.py
  extract/    layout.py  grid.py  document.py
  report/     pdf_report.py  crops.py  annotate.py
tests/        grammar, layout, grid, comparison, interface, end-to-end on two made-up projects
notebooks/    demo.ipynb (generated by build_notebook.py)
tools/        build_slides.py (presentation deck), diagnostics/ (counts-only probes used during development)
```

## Confidentiality

The project PDFs and everything derived from them (`outputs/`, OCR cache) are excluded from git by `.gitignore`.
Delete them from the workstation at the end of the event, as the rules require.

## Présentation PowerPoint : quoi mettre

La présentation vaut 10 points et consiste en 10 minutes d'exécution en direct sur le projet d'évaluation, jamais
vu. Le reste de la grille : détection des non-conformités 30 (le rappel pèse plus que la précision), extraction et
qualité du JSON 20, utilité du rapport 15, qualité technique 15, robustesse et généralisation 10. Les diapositives
servent à occuper utilement le temps pendant que l'OCR tourne. Viser 8 diapositives. Ne montrer aucun contenu des
plans confidentiels : seulement des comptes, des schémas et le projet inventé du notebook de démonstration.

### Avant de monter sur scène
- Lancer `pytest -m slow` une fois avec une connexion, pour que les modèles d'OCR soient en cache.
- Dès que le projet d'évaluation est remis : `python -m l2c_rebar run chemin\vers\EVAL --fast`.

### Diapositive 1 — Titre (15 s)
- `l2c-rebar` : des plans de structure aux dessins d'atelier. Équipe et membres.
- Une phrase : « L'outil dit à l'ingénieur où l'armature dessinée en atelier ne correspond pas au plan. »

### Diapositive 2 — Le problème (40 s)
- Comparer les plans de structure d'un bâtiment en béton armé aux dessins d'atelier du sous-traitant.
- Quatre statuts exigés : conforme, non conforme (avec l'écart), manquant, ajouté.
- Contrainte forte : documents confidentiels, tout tourne sur la machine locale, rien n'est envoyé à un service
  infonuagique ni à une IA externe.

### Diapositive 3 — Ce que l'outil produit (45 s) → utilité du rapport
- Base JSON au schéma de l'annexe A (feuillet, page, x, y, élément, barres).
- Comparaison : chaque élément classé, avec l'écart.
- Rapport PDF : comptes par feuillet, liste des constats par feuillet et axes, puis chaque non-conformité avec un
  extrait d'image des deux dessins.
- Fichier CSV des écarts, ouvrable dans Excel.
- PDF annotés : chaque constat encerclé, avec un lien vers son vis-à-vis.
- Visuel : une page du rapport du projet inventé (`notebooks/demo.ipynb`).

### Diapositive 4 — Comment ça marche (60 s) → qualité technique
- Chaîne : PDF → lignes de texte avec coordonnées → annotations d'armature → éléments → appariement →
  comparaison → rapport.
- Grammaire des annotations : `8-25M`, `12(6)-15M`, `10M@300 c/c` côté plan ; `24 15M 15A12 @12"` côté atelier.
  Espacements et longueurs convertis en millimètres.
- Repérage sur la grille : une colonne n'a pas de nom sur le plan, elle est nommée par son croisement d'axes
  (`B-3`), comme dans les dessins d'atelier.
- Appariement par nom ou croisement d'axes ; sinon par contenu dans le même type d'élément et le même étage.
- Comparaison : un plan en millimètres et un dessin en pouces concordent à 2 % près.

### Diapositive 5 — Lire les feuilles sans texte (40 s) → robustesse
Tableau des annotations lues exactement sur une page de référence :

| Réglage | Lues exactement |
|---|---|
| PP-OCRv4, 200 ppp | 57 % |
| PP-OCRv5, 200 ppp | 79 % |
| PP-OCRv5, 300 ppp (défaut) | 85 % |

- Sur les tracés vectoriels, la page est redessinée avec seulement les traits de la taille d'un caractère.
- Tout résultat qui repose sur l'OCR est marqué « à valider ».

### Diapositive 6 — Gravité, confiance et calibrage par accord (50 s) → détection, généralisation
- Gravité : critique (moins d'acier que prévu), majeur (barres introuvables, substitution, longueur plus courte),
  mineur (plus d'acier que prévu).
- Confiance entre 0 et 1 ; sous 0,60 le constat est « à valider » et l'ingénieur le confirme ou le rejette dans
  l'interface.
- Calibrage par accord : la plupart d'un dessin d'atelier concorde avec son plan, donc quand une convention est
  ouverte (noms, étages, total ou par élément), l'outil garde la lecture qui concorde le plus souvent. C'est ce qui
  permet de généraliser à un projet inconnu.

### Diapositive 7 — Démo en direct (5 à 6 min) → présentation
1. Pendant que l'OCR tourne : parcourir le rapport d'un projet de développement (liste des constats, une fiche
   avec ses deux extraits d'image, les PDF annotés et leurs liens croisés).
2. À la fin de l'exécution : ouvrir `outputs\EVAL\EVAL_rapport.pdf`.
3. `python -m l2c_rebar ui` : valider un constat incertain, le rapport se régénère.
4. Si une liste de non-conformités connues est fournie : `python -m l2c_rebar evaluate` affiche, ligne par ligne,
   jusqu'où l'outil est allé.
- Bonus à montrer si le temps le permet : comparaison de révisions (`python -m l2c_rebar diff`).

### Diapositive 8 — Limites connues et confidentialité (45 s)
- Dalles, fondations et poutres : confirmées par position, pas contredites. Sur le seul projet de développement
  fourni avec une liste de non-conformités, l'outil trouve les deux cas de colonnes et manque les cas de dalles, de
  fondations et de murs.
- L'ordre des lits d'armature n'est pas comparé.
- L'OCR reste imparfait même à 85 %.
- Aucun modèle entraîné : modèles d'OCR libres pré-entraînés (PP-OCR, Apache 2.0).
- Développé sans faire lire les dessins à l'assistant d'IA : comptes, motifs masqués et score seulement ; une
  exception déclarée dans le README.
- Les données sont supprimées des postes à la fin de l'événement.

### Questions probables du jury
- Pourquoi manquez-vous les dalles ? Seule une annotation de dalle sur cinq a son vis-à-vis assez proche pour
  comparer par position sans déclencher surtout de fausses alertes.
- Comment savoir si un constat est fiable ? Confiance, gravité et marque « à valider ».
- Et sur un projet d'un autre détailleur ? Calibrage par accord, diapositive 6.

### À ne pas faire
- Ne projeter aucun extrait des plans confidentiels dans les diapositives.
- Ne pas annoncer un taux de détection global : un seul projet avait une liste de référence.
- Ne pas promettre que les constats lus par OCR sont exacts.
