"""Build the presentation deck (PDF, 16:9) from the outputs of the four development projects.

    python tools/build_slides.py            -> outputs/presentation/l2c_rebar_presentation.pdf

The deck shows counts read from outputs/<project>/<project>_comparaison.json and, when present,
the stage flags of outputs/CLP/CLP_evaluation.json.  It never shows drawing content, so it can be
projected; it is written under outputs/ (not committed) because it names the projects.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

W, H = 960.0, 540.0
M = 54.0
INK = colors.HexColor("#1E2327")
RUST = colors.HexColor("#B9481C")
STEEL = colors.HexColor("#56697A")
CONCRETE = colors.HexColor("#EEF0F2")
LINE = colors.HexColor("#C9D0D7")
OK = colors.HexColor("#2E7D32")
BAD = colors.HexColor("#C62828")
WARN = colors.HexColor("#D9822B")
WHITE = colors.white

PROJECTS = ("CLP", "LIGREP", "WP2", "EspCa3B")


def style(size: float, color=INK, bold: bool = False, leading: float | None = None) -> ParagraphStyle:
    return ParagraphStyle("s", fontName="Helvetica-Bold" if bold else "Helvetica", fontSize=size,
                          leading=leading or size * 1.28, textColor=color, alignment=TA_LEFT)


def text(c: canvas.Canvas, markup: str, x: float, top: float, width: float, st: ParagraphStyle) -> float:
    """Draw wrapped text with its top at `top`; returns the bottom y."""
    p = Paragraph(markup, st)
    _, h = p.wrap(width, H)
    p.drawOn(c, x, top - h)
    return top - h


def title(c: canvas.Canvas, label: str, kicker: str = "") -> None:
    if kicker:
        c.setFillColor(RUST)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(M, H - 50, kicker.upper())
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 30)
    c.drawString(M, H - 88, label)


def footer(c: canvas.Canvas, n: int, dark: bool = False) -> None:
    c.setFillColor(colors.HexColor("#8A98A6") if dark else STEEL)
    c.setFont("Helvetica", 9)
    c.drawString(M, 22, "l2c-rebar · vérification d'armature plan / atelier · tout s'exécute en local")
    c.drawRightString(W - M, 22, str(n))


def card(c: canvas.Canvas, x: float, y: float, w: float, h: float, fill=CONCRETE) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(fill)
    c.roundRect(x, y, w, h, 8, stroke=0, fill=1)


def dot(c: canvas.Canvas, x: float, y: float, color, r: float = 6) -> None:
    c.setFillColor(color)
    c.circle(x, y, r, stroke=0, fill=1)


def arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, color=STEEL) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.6)
    c.line(x1, y1, x2 - 6, y2)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - 8, y2 + 4.5)
    p.lineTo(x2 - 8, y2 - 4.5)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def load_stats(root: Path) -> dict[str, dict]:
    out = {}
    for name in PROJECTS:
        path = root / name / f"{name}_comparaison.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            out[name] = {"stats": data["statistiques"], "couverture": data.get("couverture", {})}
    return out


def agreement_text(project: dict) -> str:
    share = project["couverture"].get("accord_par_type", {}).get("colonne")
    return "-" if share is None else f"{round(100 * share)} %"


def fmt(n: int) -> str:
    return f"{n:,}".replace(",", " ")


# ------------------------------------------------------------------ slides

def s_cover(c, n):
    c.setFillColor(INK)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(RUST)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(M, H - 150, "CODEML 2026 · DÉFI L2C · DU PLAN AUX DESSINS D'ATELIER")
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 64)
    c.drawString(M, H - 230, "l2c-rebar")
    text(c, "Les dessins d'atelier d'armature vérifiés contre les plans de structure : "
            "lus, appariés, comparés, et chaque écart situé sur le feuillet.",
         M, H - 262, 640, style(20, colors.HexColor("#D5DCE2"), leading=27))
    # rebar motif: a row of bar sections
    for i in range(9):
        dot(c, M + 9 + i * 26, 92, RUST if i % 4 == 0 else colors.HexColor("#3A4249"), 8)
    c.setFillColor(colors.HexColor("#8A98A6"))
    c.setFont("Helvetica", 11)
    c.drawString(M, 58, "Démonstration en direct sur le projet d'évaluation")
    footer(c, n, dark=True)


def s_problem(c, n, stats):
    title(c, "Une revue manuelle, feuillet par feuillet", "Le problème")
    y = text(c, "Avant fabrication, l'ingénieur compare les PDF côte à côte : repère, diamètre, quantité, "
                "espacement, longueur de chaque barre. Des centaines de pages, aucune valeur ajoutée, "
                "et une erreur qui passe inaperçue se paie au chantier.", M, H - 120, 470, style(16, leading=23))
    text(c, "<b>Notre réponse</b> : un outil qui lit les deux jeux de documents, apparie chaque élément du plan "
            "à son équivalent d'atelier et ne montre à l'ingénieur que ce qui diffère. La décision reste la sienne.",
         M, y - 18, 470, style(16, leading=23))
    plan = sum(s["stats"].get("plan_pages", 0) for s in stats.values())
    shop = sum(s["stats"].get("atelier_pages", 0) for s in stats.values())
    ocr = sum(s["stats"].get("atelier_pages_ocr", 0) for s in stats.values())
    bars = sum(s["stats"].get("plan_barres", 0) + s["stats"].get("atelier_barres", 0) for s in stats.values())
    x0 = 580
    for i, (value, label) in enumerate(((fmt(plan + shop), f"pages lues sur {len(stats)} projets"),
                                        (fmt(ocr), "pages d'atelier sans texte, lues par OCR local"),
                                        (fmt(bars), "annotations d'armature extraites"))):
        top = H - 130 - i * 112
        card(c, x0, top - 92, W - M - x0, 92)
        c.setFillColor(RUST if i == 0 else INK)
        c.setFont("Helvetica-Bold", 38)
        c.drawString(x0 + 22, top - 50, value)
        c.setFillColor(STEEL)
        c.setFont("Helvetica", 12.5)
        c.drawString(x0 + 22, top - 74, label)
    footer(c, n)


def s_pipeline(c, n):
    title(c, "Du PDF au rapport, en six étapes", "Architecture")
    steps = [
        ("Lire", "Texte vectoriel du PDF ; sinon OCR local (PP-OCRv5, GPU), tuile par tuile"),
        ("Analyser", "Grammaire des annotations : 8-25M, 12(6)-15M, 15M@300, 24 15M 15A12 @12\""),
        ("Regrouper", "Annotation → élément : repère, ligne de tableau, croisement d'axes"),
        ("Apparier", "Par nom ou croisement d'axes, sinon par position sur la grille, sinon par contenu"),
        ("Comparer", "Quantité, diamètre, espacement, longueur ; écarts classés par gravité"),
        ("Rapporter", "PDF par feuillet, JSON Annexe A, PDF annotés, interface de validation"),
    ]
    bw, gap, top = 132.0, 18.0, H - 150
    for i, (name, desc) in enumerate(steps):
        x = M + i * (bw + gap)
        card(c, x, top - 180, bw, 180, CONCRETE if i % 5 else colors.HexColor("#F6E7E0"))
        c.setFillColor(RUST)
        c.setFont("Helvetica-Bold", 26)
        c.drawString(x + 14, top - 40, str(i + 1))
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(x + 14, top - 66, name)
        text(c, desc, x + 14, top - 80, bw - 26, style(11.5, STEEL, leading=15))
        if i < len(steps) - 1:
            arrow(c, x + bw + 2, top - 90, x + bw + gap - 1, top - 90)
    text(c, "Python · PyMuPDF · RapidOCR / onnxruntime-directml · pydantic · ReportLab · Streamlit. "
            "Aucun document, texte ou image ne quitte la machine.", M, 112, W - 2 * M, style(13, STEEL))
    footer(c, n)


def s_columns(c, n):
    title(c, "Les colonnes : retrouvées par leur croisement d'axes", "Idée clé")
    # made-up illustration of a column plan
    gx0, gy0, step = M + 40, 128, 108
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.setDash(6, 4)
    for i in range(3):
        c.line(gx0 + i * step, gy0 - 30, gx0 + i * step, gy0 + 2 * step + 30)
        c.line(gx0 - 30, gy0 + i * step, gx0 + 2 * step + 30, gy0 + i * step)
    c.setDash()
    for i, lab in enumerate("123"):
        c.setStrokeColor(STEEL)
        c.setFillColor(WHITE)
        c.circle(gx0 + i * step, gy0 + 2 * step + 46, 13, stroke=1, fill=1)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 12)
        c.drawCentredString(gx0 + i * step, gy0 + 2 * step + 42, lab)
    for i, lab in enumerate("CBA"):
        c.setFillColor(WHITE)
        c.circle(gx0 - 46, gy0 + i * step, 13, stroke=1, fill=1)
        c.setFillColor(INK)
        c.drawCentredString(gx0 - 46, gy0 + i * step - 4, lab)
    for i in range(3):
        for j in range(3):
            c.setFillColor(RUST if (i, j) == (1, 1) else INK)
            c.rect(gx0 + i * step - 7, gy0 + j * step - 7, 14, 14, stroke=0, fill=1)
    tx, ty = gx0 + step + 18, gy0 + step + 22
    card(c, tx, ty, 96, 40, colors.HexColor("#F6E7E0"))
    c.setFillColor(INK)
    c.setFont("Helvetica", 10)
    c.drawString(tx + 8, ty + 25, "ARM.: 8-25M")
    c.drawString(tx + 8, ty + 10, "LIG.: 10M@300")
    c.setStrokeColor(RUST)
    c.setLineWidth(1.2)
    c.line(gx0 + step + 7, gy0 + step + 7, tx, ty + 10)
    c.setFillColor(STEEL)
    c.setFont("Helvetica-Oblique", 10)
    c.drawString(gx0 - 60, 66, "Illustration inventée, pas un extrait de plan.")
    x = 520
    y = text(c, "Au plan, une colonne n'a pas de nom : c'est un rectangle plein à un croisement de la grille, "
                "avec une étiquette posée à côté.", x, H - 130, W - M - x, style(15, leading=21))
    y = text(c, "L'outil reconstruit la grille à partir des bulles, relie chaque étiquette à son rectangle par "
                "le décalage que le dessinateur répète d'une colonne à l'autre, et nomme la colonne "
                "<b>B-2</b>, comme le fait le dessin d'atelier.", x, y - 14, W - M - x, style(15, leading=21))
    text(c, "Résultat : sur les projets dont l'atelier a une couche texte, la quasi-totalité des colonnes "
            "appariées concordent, et les deux non-conformités de colonnes de la liste connue sont trouvées "
            "avec les bonnes valeurs.", x, y - 14, W - M - x, style(15, RUST, leading=21))
    footer(c, n)


def s_calibration(c, n):
    title(c, "Calibré par accord, pas par hypothèses", "Idée clé")
    text(c, "Un dessin d'atelier concorde en grande partie avec son plan. Quand une convention est ambiguë, l'outil "
            "essaie les lectures possibles et garde celle qui confirme le plus souvent le plan.",
         M, H - 112, W - 2 * M, style(15, STEEL, leading=21))
    items = [
        ("Étages", "« NIV 2@3 » détaille-t-il l'étage 2 ou 3 ? Correspondance des étages dans l'ordre, par accord."),
        ("Tableau ou détails", "Une feuille de repères se lit en tableau ou en détails : la meilleure lecture gagne, par type."),
        ("Par élément ou total", "Un détail commun « B-12, B-13 » donne les barres d'un élément ou le total."),
        ("Confiance dans les noms", "Si moins de la moitié des barres appariées par nom concordent, l'appariement est abandonné."),
        ("Lignes OCR recollées", "Les morceaux d'une ligne lus séparément sont recollés là où cela confirme davantage."),
        ("Couverture", "Un feuillet mal couvert par l'atelier est signalé une fois, pas élément par élément."),
    ]
    cw, ch, gap = (W - 2 * M - 2 * 18) / 3, 132, 18
    for i, (head, body) in enumerate(items):
        x = M + (i % 3) * (cw + gap)
        top = H - 175 - (i // 3) * (ch + gap)
        card(c, x, top - ch, cw, ch)
        dot(c, x + 22, top - 26, RUST, 7)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(x + 38, top - 31, head)
        text(c, body, x + 18, top - 48, cw - 36, style(12.5, INK, leading=17))
    footer(c, n)


def s_results(c, n, stats):
    title(c, "Résultats sur les quatre projets fournis", "Résultats")
    head = ["Projet", "Pages plan", "Pages atelier", "dont OCR", "Éléments", "Conformes", "Non conformes",
            "Manquants", "Ajoutés", "Accord"]
    widths = [92, 76, 92, 68, 82, 82, 104, 82, 66, 108]
    x0, top, rh = M, H - 130, 40
    card(c, x0, top - rh, sum(widths), rh, INK)
    x = x0
    for h, w in zip(head, widths):
        c.setFillColor(WHITE)
        c.setFont("Helvetica-Bold", 11.5)
        c.drawString(x + 10, top - 25, h)
        x += w
    for r, (name, s) in enumerate(stats.items()):
        st = s["stats"]
        y = top - rh * (r + 2)
        if r % 2 == 0:
            card(c, x0, y, sum(widths), rh, CONCRETE)
        values = [name, st.get("plan_pages", 0), st.get("atelier_pages", 0), st.get("atelier_pages_ocr", 0),
                  st.get("plan_elements", 0) + st.get("atelier_elements", 0), st.get("conforme", 0),
                  st.get("non_conforme", 0), st.get("manquant_atelier", 0), st.get("ajoute_atelier", 0),
                  agreement_text(s)]
        x = x0
        for k, (v, w) in enumerate(zip(values, widths)):
            c.setFillColor(BAD if k == 6 else INK)
            c.setFont("Helvetica-Bold" if k in (0, 6) else "Helvetica", 13)
            c.drawString(x + 10, y + 14, fmt(v) if isinstance(v, int) else str(v))
            x += w
    text(c, "Accord : part des barres des colonnes appariées par croisement d'axes que l'atelier confirme ; "
            "sous 90 %, les écarts de colonnes du projet passent « à valider ». "
            "Chaque résultat porte une confiance ; sous 0,60 il est marqué « à valider » et l'interface permet à "
            "l'ingénieur de le confirmer ou de le rejeter. Les non-conformités sont classées par gravité : "
            "critique (moins d'acier), majeur (barres introuvables, diamètre plus gros, longueur plus courte), "
            "mineur (surplus).", M, top - rh * (len(stats) + 1) - 30, W - 2 * M, style(13, STEEL, leading=18))
    footer(c, n)


def s_known(c, n, evaluation):
    title(c, "Sur la liste de non-conformités connues (CLP)", "Mesure honnête")
    if not evaluation:
        text(c, "Lancer <font face='Courier'>python -m l2c_rebar evaluate CLP CLP/CLP_dismatch.xlsx</font> pour "
                "remplir cette diapositive.", M, H - 130, W - 2 * M, style(15))
        footer(c, n)
        return
    summary, rows = evaluation["resume"], evaluation["lignes"]
    card(c, M, H - 330, 250, 200, INK)
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 64)
    c.drawString(M + 24, H - 225, f"{summary['signalees']}/{summary['connues']}")
    text(c, f"trouvées au bon endroit, dont {summary['memes_valeurs']} avec les mêmes valeurs plan et atelier",
         M + 24, H - 240, 205, style(13.5, colors.HexColor("#D5DCE2"), leading=18))
    x = M + 290
    step = min(44.0, 250.0 / max(len(rows), 1))  # the rows must end above the closing paragraph
    for i, r in enumerate(rows):
        y = H - 140 - i * step
        dot(c, x + 8, y - 6, OK if r["trouvee"] else (WARN if "bon endroit" in r["etape"] else BAD), 7)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 13.5)
        c.drawString(x + 26, y - 11, r["type_element"].capitalize())
        c.setFont("Helvetica", 13)
        c.setFillColor(STEEL)
        c.drawString(x + 170, y - 11, r["etape"])
    text(c, "Ce qui manque, et pourquoi : la valeur d'une dalle est écrite « 12(6) » au plan et détaillée autrement à "
            "l'atelier ; aucune étiquette ne relie la semelle à son détail ; aucun dessin "
            "d'atelier de mur n'est fourni pour CLP. Chaque cas est mesuré, pas deviné.",
         M, 118, W - 2 * M, style(13, STEEL, leading=18))
    footer(c, n)


def s_deliverables(c, n):
    title(c, "Ce que l'ingénieur reçoit", "Livrables")
    tiles = [
        ("Rapport PDF par feuillet", "Conformes et non conformes par feuillet, puis une fiche par écart : valeurs plan "
                                     "et atelier, gravité, confiance, axes (B-12), extraits d'image des deux dessins."),
        ("Base JSON, Annexe A", "Chaque annotation du plan et de l'atelier : feuillet, page, x, y au centre, élément, "
                                "repère, diamètre, quantité, espacement, longueur. Validée par pydantic."),
        ("PDF annotés", "Les écarts encerclés directement sur le plan et sur le dessin d'atelier, avec un lien "
                        "cliquable de l'un à l'autre."),
        ("Interface de validation", "Streamlit : l'ingénieur confirme ou rejette les cas incertains, le rapport est "
                                    "régénéré. Plus : comparaison de deux révisions d'un dessin d'atelier."),
    ]
    cw, ch, gap = (W - 2 * M - 18) / 2, 136, 18
    for i, (head, body) in enumerate(tiles):
        x = M + (i % 2) * (cw + gap)
        top = H - 125 - (i // 2) * (ch + gap)
        card(c, x, top - ch, cw, ch)
        c.setFillColor(RUST)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(x + 20, top - 30, f"0{i + 1}")
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 17)
        c.drawString(x + 52, top - 31, head)
        text(c, body, x + 20, top - 48, cw - 40, style(13, INK, leading=18))
    text(c, "Code : CLI <font face='Courier'>python -m l2c_rebar run PROJET</font>, tests automatisés sur des projets "
            "inventés, notebook de démonstration, README (architecture, hypothèses, limites).",
         M, 84, W - 2 * M, style(12.5, STEEL))
    footer(c, n)


def s_limits(c, n):
    title(c, "Limites, en toute transparence", "Limites")
    items = [
        ("Dalles", "Au plan « 12(6) » près d'une barre ; à l'atelier, la même armature est détaillée autrement. "
                   "Seule une annotation sur cinq a son jumeau au même endroit : la position confirme, elle ne "
                   "contredit pas, pour ne pas noyer l'ingénieur de fausses alertes."),
        ("Fondations", "Une semelle sans étiquette : le lien entre l'emplacement et son détail n'est pas lu automatiquement."),
        ("OCR", "85 % des annotations lues exactement sur la page de référence ; tout résultat issu de l'OCR le dit."),
        ("Mise en page", "Les règles de rattachement annotation → élément sont des heuristiques, calibrées sur les "
                         "quatre projets fournis."),
        ("Confidentialité", "L'outil a été écrit avec un assistant de code IA qui n'a jamais lu les documents : "
                            "il a travaillé sur des comptes et des motifs masqués (chiffres → 9, lettres → A)."),
    ]
    top = H - 120
    for head, body in items:
        dot(c, M + 7, top - 9, RUST, 5)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 15)
        c.drawString(M + 22, top - 14, head)
        top = text(c, body, M + 170, top - 1, W - 2 * M - 170, style(13.5, INK, leading=18)) - 16
    footer(c, n)


def s_demo(c, n):
    c.setFillColor(INK)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(RUST)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(M, H - 60, "DÉMONSTRATION")
    c.setFillColor(WHITE)
    c.setFont("Helvetica-Bold", 34)
    c.drawString(M, H - 104, "Le projet d'évaluation, en direct")
    card(c, M, H - 330, W - 2 * M, 190, colors.HexColor("#2A3137"))
    lines = [
        "python -m l2c_rebar run PROJET_EVAL --fast        # JSON + rapport + PDF annotés",
        "python -m l2c_rebar ui                            # validation des cas incertains",
        "python -m l2c_rebar diff ancienne.pdf nouvelle.pdf  # comparaison de révisions",
    ]
    c.setFont("Courier", 15)
    for i, line in enumerate(lines):
        c.setFillColor(colors.HexColor("#E8C9BB") if i == 0 else colors.HexColor("#C7D0D8"))
        c.drawString(M + 24, H - 185 - i * 40, line)
    text(c, "Durée : quelques secondes par feuillet à couche texte ; 10 à 25 secondes par page sans texte (OCR sur "
            "GPU), avec le temps restant affiché. Une page illisible est sautée, jamais bloquante.",
         M, H - 356, W - 2 * M, style(15, colors.HexColor("#D5DCE2"), leading=21))
    footer(c, n, dark=True)


def build(out_path: Path, outputs: Path) -> Path:
    stats = load_stats(outputs)
    ev_path = outputs / "CLP" / "CLP_evaluation.json"
    evaluation = json.loads(ev_path.read_text(encoding="utf-8")) if ev_path.exists() else None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=(W, H))
    c.setTitle("l2c-rebar - présentation")
    slides = [lambda n: s_cover(c, n), lambda n: s_problem(c, n, stats), lambda n: s_pipeline(c, n),
              lambda n: s_columns(c, n), lambda n: s_calibration(c, n), lambda n: s_results(c, n, stats),
              lambda n: s_known(c, n, evaluation), lambda n: s_deliverables(c, n), lambda n: s_limits(c, n),
              lambda n: s_demo(c, n)]
    for i, slide in enumerate(slides, start=1):
        slide(i)
        c.showPage()
    c.save()
    return out_path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "outputs" / "presentation" / "l2c_rebar_presentation.pdf"
    print(build(target, root / "outputs"))
