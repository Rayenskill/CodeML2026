"""Command line entry point.

    python -m l2c_rebar run <dossier_projet> [--out outputs/<projet>]
    python -m l2c_rebar evaluate <dossier_projet> <ecarts_connus.xlsx>
    python -m l2c_rebar diff <ancienne_rev.pdf> <nouvelle_rev.pdf>
    python -m l2c_rebar ui

The console only ever shows counts and file paths, never drawing content.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import __version__
from .config import Config
from .models import AJOUTE, CONFORME, MANQUANT, NON_CONFORME


def _config(args: argparse.Namespace) -> Config:
    cfg = Config.load(getattr(args, "config", None))
    for name in ("ocr", "workers"):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if getattr(args, "fast", False):
        # Live demonstration: OCR at 200 dpi (about 25% faster, 79% of callouts read exactly instead of 85%)
        # and no image extracts in the report.
        cfg.ocr_dpi = 200
        cfg.crops = False
    if getattr(args, "no_crops", False):
        cfg.crops = False
    if getattr(args, "no_annotate", False):
        cfg.annotate = False
    return cfg


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_project
    from .report.annotate import annotate_project
    from .report.pdf_report import build_report

    cfg = _config(args)
    project = Path(args.project)
    out_dir = Path(args.out) if args.out else Path("outputs") / project.name
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"  {m}", flush=True))

    print(f"Projet {project.name} -> {out_dir}")
    out = run_project(project, out_dir, cfg, progress=say)

    say("Rapport PDF")
    report = build_report(out.project, out.results, out.coverage, out.stats, out_dir / f"{out.project}_rapport.pdf", cfg)
    out.files["rapport"] = report
    if cfg.annotate:
        say("PDF annotés")
        annotated = annotate_project(out.results, out_dir)
        out.stats["pdf_annotes"] = len(annotated)

    s = out.stats
    print(f"  pages : plan {s.get('plan_pages', 0)}, atelier {s.get('atelier_pages', 0)} "
          f"(OCR {s.get('atelier_pages_ocr', 0)}, illisibles {s.get('atelier_pages_vides', 0)})")
    print(f"  éléments : plan {s['plan_elements']} ({s['plan_reperes']} avec repère), "
          f"atelier {s['atelier_elements']} ({s['atelier_reperes']} avec repère), repères communs {s['reperes_communs']}")
    print(f"  résultats : {s[CONFORME]} conformes, {s[NON_CONFORME]} non conformes, {s[MANQUANT]} manquants, "
          f"{s[AJOUTE]} ajoutés, {s['a_valider']} à valider ({s['apparies_par_repere']} appariés par repère)")
    print(f"  durée : {s['duree_s']} s")
    for name, path in out.files.items():
        print(f"  {name:12s} {path}")
    if cfg.annotate:
        print(f"  {'annotes':12s} {out_dir / 'annotes'} ({out.stats.get('pdf_annotes', 0)} fichiers)")
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    from .revisions import diff_revisions

    cfg = _config(args)
    changes = diff_revisions(Path(args.old), Path(args.new), cfg)
    counts = {k: len(v) for k, v in changes.items()}
    if args.out:
        Path(args.out).write_text(json.dumps(changes, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"Écrit : {args.out}")
    print(f"Révisions : {counts['ajoutes']} ajoutés, {counts['retires']} retirés, {counts['modifies']} modifiés, "
          f"{counts['inchanges']} inchangés")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Score a run against a spreadsheet of documented non-conformities.

    Prints stage flags per row and totals; never the content of the spreadsheet.
    """
    from .evaluate import axes_of_plan, load_known, score_run, summarize_scores
    from .extract.document import read_document
    from .models import SOURCE_PLAN
    from .pipeline import discover, run_project

    cfg = _config(args)
    project = Path(args.project)
    out = run_project(project, Path(args.out) if args.out else Path("outputs") / project.name, cfg)
    axes = {}
    for plan in discover(project).plans:
        pages, _ = read_document(plan, SOURCE_PLAN, cfg, None)
        axes.update(axes_of_plan(pages))
    scores = score_run(load_known(Path(args.known)), out.plan_elements, out.results, axes)
    summary = summarize_scores(scores)
    print(f"Non-conformités connues : {summary['connues']} | signalées au bon endroit : {summary['signalees']} | "
          f"avec les mêmes valeurs : {summary['memes_valeurs']}")
    rows = []
    for score in scores:
        if score.same_values:
            stage = "trouvée, mêmes valeurs"
        elif score.found:
            stage = "trouvée, valeurs différentes"
        elif score.plan_value_at_location:
            stage = "valeur du plan extraite au bon endroit, écart non signalé"
        elif score.plan_value_on_sheet:
            stage = "valeur du plan extraite ailleurs sur le feuillet"
        elif score.sheet_read:
            stage = "valeur du plan non extraite"
        else:
            stage = "feuillet non lu"
        print(f"  ligne {score.row} ({score.type_element}) : {stage}")
        rows.append({"ligne": score.row, "type_element": score.type_element, "etape": stage, "trouvee": score.found})
    # Stage flags only, never the content of the list.
    report = out.out_dir / f"{out.project}_evaluation.json"
    report.write_text(json.dumps({"resume": summary, "lignes": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  Écrit : {report}")
    return 0


def cmd_ui(_args: argparse.Namespace) -> int:
    app = Path(__file__).with_name("app.py")
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l2c_rebar", description="Vérification d'armature : plans vs dessins d'atelier")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Analyse un projet et produit le JSON, le rapport PDF et les PDF annotés")
    run.add_argument("project", help="Dossier du projet (plan PDF à la racine, dessins d'atelier dans les sous-dossiers)")
    run.add_argument("--out", help="Dossier de sortie (défaut : outputs/<projet>)")
    run.add_argument("--config", help="Fichier JSON de configuration")
    run.add_argument("--ocr", choices=["auto", "off", "force"], help="OCR local des pages sans texte (défaut : auto)")
    run.add_argument("--workers", type=int, help="Processus OCR en parallèle")
    run.add_argument("--no-crops", action="store_true", help="Rapport sans extraits d'image (plus rapide)")
    run.add_argument("--fast", action="store_true",
                     help="Démonstration en direct : OCR à 200 dpi et rapport sans extraits d'image (environ 25 %% plus rapide)")
    run.add_argument("--no-annotate", action="store_true", help="Ne pas produire les PDF annotés")
    run.add_argument("--quiet", action="store_true")
    run.set_defaults(func=cmd_run)

    diff = sub.add_parser("diff", help="Compare deux révisions d'un même dessin d'atelier")
    diff.add_argument("old")
    diff.add_argument("new")
    diff.add_argument("--out", help="Fichier JSON des changements")
    diff.add_argument("--config")
    diff.add_argument("--ocr", choices=["auto", "off", "force"])
    diff.set_defaults(func=cmd_diff)

    evaluate = sub.add_parser("evaluate", help="Note une analyse contre une liste d'écarts connus (fichier Excel)")
    evaluate.add_argument("project")
    evaluate.add_argument("known", help="Classeur : feuillet | emplacement sur le quadrillage | plan | dessin d'atelier")
    evaluate.add_argument("--out")
    evaluate.add_argument("--config")
    evaluate.add_argument("--ocr", choices=["auto", "off", "force"])
    evaluate.set_defaults(func=cmd_evaluate)

    ui = sub.add_parser("ui", help="Interface Streamlit (analyse et validation des cas incertains)")
    ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
