"""Builds notebooks/demo.ipynb and executes it on the synthetic project.

The notebook is generated rather than hand-edited so that it stays in step with
the code; run `python notebooks/build_notebook.py` after changing the pipeline.
"""
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# L2C - Du plan aux dessins d'atelier : exploration et démonstration\n\n"
       "This notebook walks through the pipeline end to end on one project.\n\n"
       "It runs on a **synthetic project** (`l2c_rebar.synthetic`) so that no confidential drawing content is "
       "stored in the notebook outputs. To run it on a real project, set `PROJECT` in the next cell to the "
       "project folder and re-run everything locally."),
    code("from pathlib import Path\nimport json, tempfile\nimport pymupdf, pandas as pd\n"
         "from l2c_rebar.synthetic import build_project, PLANTED\n\n"
         "WORK = Path(tempfile.mkdtemp(prefix='l2c_demo_'))\n"
         "PROJECT = build_project(WORK)          # or: Path('C:/path/to/a/real/project')\n"
         "OUT = WORK / 'out'\nPROJECT"),
    md("## 1. Data exploration\n\nA project is one plan PDF at its root and shop drawings (`DA/`) grouped by element type. "
       "The first question for every page: does it carry real text, or only drawn strokes that need OCR?"),
    code("from l2c_rebar.pipeline import discover, shop_file_hints\n\nfiles = discover(PROJECT)\nrows = []\n"
         "for path in files.plans + files.shops:\n    with pymupdf.open(path) as doc:\n"
         "        words = [len(p.get_text('words')) for p in doc]\n"
         "        hint = ('plan', ()) if path in files.plans else shop_file_hints(path, files.root)\n"
         "        rows.append({'fichier': path.name, 'pages': doc.page_count, 'mots': sum(words),\n"
         "                     'pages_sans_texte': sum(w < 25 for w in words), 'type': hint[0], 'niveaux': hint[1]})\n"
         "pd.DataFrame(rows)"),
    md("Text lines come out in *displayed* page coordinates (top-left origin, PDF points), which is what Appendix A asks for."),
    code("from l2c_rebar.pdf.text import text_layer_lines\n\nwith pymupdf.open(files.plans[0]) as doc:\n"
         "    lines = text_layer_lines(doc[3])\n"
         "pd.DataFrame([{'texte': l.text, 'x': round((l.bbox[0]+l.bbox[2])/2, 1), 'y': round((l.bbox[1]+l.bbox[3])/2, 1)} for l in lines]).head(12)"),
    md("## 2. Reading a callout\n\nThe grammar covers the notations met on plans and shop drawings, metric or imperial spacing."),
    code("from l2c_rebar.parsing.rebar import find_bars\n\nsamples = ['VERT.: 8-25M', 'LIG.: 10M@300 c/c', 'LIG.: 10M@12\" c/c', '12(6)-15M',\n"
         "           'T: 24 15M 15A12 @12\"', 'B: 24 15M 20-6', '8-20M x 3000 SUP.']\n"
         "pd.DataFrame([{'texte': s, **{k: getattr(b, k) for k in ('quantite', 'diametre', 'espacement_mm', 'longueur_mm', 'repere', 'role')}}\n"
         "              for s in samples for b in find_bars(s)])"),
    md("## 3. Full pipeline: extract, build the JSON database, compare"),
    code("from l2c_rebar.config import Config\nfrom l2c_rebar.pipeline import run_project\n\n"
         "out = run_project(PROJECT, OUT, Config(ocr='auto'))\n"
         "{k: v for k, v in out.stats.items() if k != 'par_type'}"),
    code("pd.DataFrame(out.stats['par_type']).T.fillna(0).astype(int)"),
    md("The JSON database follows Appendix A exactly (validated with pydantic):"),
    code("records = json.loads(out.files['elements'].read_text(encoding='utf-8'))\nprint(len(records), 'records')\n"
         "print(json.dumps(next(r for r in records if r['element'] == 'C3' and r['source'] == 'plan'), indent=2, ensure_ascii=False))"),
    md("## 4. Findings\n\nEach element is classified: compliant, non-compliant (with the discrepancy), missing from the shop "
       "drawings, or added in the shop drawings."),
    code("from l2c_rebar.compare import summarize\n\npd.DataFrame(summarize(out.results)).T"),
    code("pd.DataFrame([{'id': r.id, 'feuillet': r.feuillet, 'type': r.type_element, 'element': r.element, 'statut': r.statut,\n"
         "               'gravite': r.gravite if r.ecarts else '', 'confiance': r.confiance, 'ecart': ' ; '.join(e.message for e in r.ecarts)}\n"
         "              for r in out.results if r.statut != 'conforme'])"),
    md("The synthetic project plants five discrepancies; all five must be found, with no false alarm:"),
    code("found = {(r.type_element, e.attribut, e.plan, e.atelier) for r in out.results for e in r.ecarts}\n"
         "pd.DataFrame([{**p, 'detecte': (p['type'], p['attribut'], p['plan'], p['atelier']) in found} for p in PLANTED])"),
    md("## 5. PDF report and annotated PDFs\n\nThe report opens with counts per plan sheet, then one list of every "
       "non-conformity and missing element, by sheet and grid place (`axes B-12`), then a detailed card per finding "
       "with image extracts of both drawings."),
    code("from l2c_rebar.report.pdf_report import build_report\nfrom l2c_rebar.report.annotate import annotate_project\n\n"
         "report = build_report(out.project, out.results, out.coverage, out.stats, OUT / 'rapport.pdf', Config())\n"
         "annotated = annotate_project(out.results, OUT)\nprint(report.name, '+', len(annotated), 'annotated PDFs')"),
    code("from IPython.display import Image, display\n\nwith pymupdf.open(report) as doc:\n"
         "    for i in (0, 1):\n        display(Image(data=doc[i].get_pixmap(dpi=80).tobytes('png')))"),
    code("plan = next(p for src, p in annotated.items() if 'PLAN' in src)\nwith pymupdf.open(plan) as doc:\n"
         "    display(Image(data=doc[3].get_pixmap(dpi=60).tobytes('png')))"),
    md("## 6. Columns located by their place on the grid\n\nOn a column plan a column has no name: it is a filled "
       "rectangle at a grid crossing, with a tag beside it. The grid is rebuilt from its bubbles, each tag is paired "
       "with its rectangle, and the column is named by its crossing (`B-3`), which is how shop drawings refer to it. "
       "A second made-up project shows this."),
    code("from l2c_rebar.synthetic import build_grid_project, write_known_list, GRID_PLANTED\n\n"
         "GRID = build_grid_project(WORK)\n"
         "with pymupdf.open(discover(GRID).plans[0]) as doc:\n"
         "    display(Image(data=doc[0].get_pixmap(dpi=60).tobytes('png')))"),
    code("grid_out = run_project(GRID, WORK / 'out_grid', Config(ocr='off'))\n"
         "pd.DataFrame([{'feuillet': e.feuillet, 'colonne': e.element, 'niveau': e.levels, 'x': round(e.x), 'y': round(e.y),\n"
         "               'armature': [b.describe() for b in e.bars]} for e in grid_out.plan_elements[:8]])"),
    md("The shop drawings name storeys their own way (`RDC@2`, `2@3`) and draw one detail for two identical columns, "
       "with totals. Both are resolved by keeping the reading that agrees most often with the plan."),
    code("print(grid_out.coverage.level_shift, grid_out.coverage.lecture)\n"
         "pd.DataFrame([{'id': r.id, 'feuillet': r.feuillet, 'colonne': r.element, 'statut': r.statut, 'gravite': r.gravite if r.ecarts else '',\n"
         "               'ecart': ' ; '.join(e.message for e in r.ecarts), 'note': r.note}\n"
         "              for r in grid_out.results if r.statut != 'conforme' or r.note])"),
    md("### Scoring a run against documented non-conformities\n\nThe jury holds a list of known non-conformities: sheet, "
       "grid location, what the plan says, what the shop drawing says. `evaluate` scores a run against such a list and "
       "says, for each miss, at which stage it was lost."),
    code("from l2c_rebar.evaluate import axes_of_plan, load_known, score_run, summarize_scores\n"
         "from l2c_rebar.extract.document import read_document\n\n"
         "known = load_known(write_known_list(WORK / 'known.xlsx', GRID_PLANTED))\n"
         "pages, _ = read_document(discover(GRID).plans[0], 'plan', Config(ocr='off'), None)\n"
         "summarize_scores(score_run(known, grid_out.plan_elements, grid_out.results, axes_of_plan(pages)))"),
    md("## 7. Same thing from the command line\n\n```\npython -m l2c_rebar run <project folder>               # JSON + PDF report + annotated PDFs\n"
       "python -m l2c_rebar run <project folder> --fast        # live demonstration: OCR at 200 dpi, no image extracts\n"
       "python -m l2c_rebar evaluate <project folder> known.xlsx  # score against documented non-conformities\n"
       "python -m l2c_rebar ui                                 # interface with validation of uncertain cases\n"
       "python -m l2c_rebar diff old.pdf new.pdf               # what changed between two revisions\n```"),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
NotebookClient(nb, timeout=600, kernel_name="python3").execute()
target = Path(__file__).with_name("demo.ipynb")
nbf.write(nb, target)
print("written", target, "cells", len(nb.cells))
