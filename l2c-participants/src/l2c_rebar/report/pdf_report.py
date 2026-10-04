"""The PDF discrepancy report, organised by plan sheet (ReportLab)."""

from __future__ import annotations

import io
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .. import __version__
from ..compare import Coverage, summarize
from ..config import Config
from ..models import AJOUTE, CONFORME, MANQUANT, NON_CONFORME, Element, Result
from .crops import Cropper

GREEN = colors.HexColor("#2E7D32")
RED = colors.HexColor("#C62828")
ORANGE = colors.HexColor("#E07000")
BLUE = colors.HexColor("#1F66C1")
GREY = colors.HexColor("#5F6B7A")
LIGHT = colors.HexColor("#F3F5F8")
RULE = colors.HexColor("#C9D0D9")

STATUS_LABEL = {CONFORME: "Conformes", NON_CONFORME: "Non conformes", MANQUANT: "Manquants à l'atelier",
                AJOUTE: "Ajoutés à l'atelier"}
STATUS_COLOR = {CONFORME: GREEN, NON_CONFORME: RED, MANQUANT: ORANGE, AJOUTE: BLUE}
GRAVITE_COLOR = {"critique": RED, "majeur": ORANGE, "mineur": GREY}

_base = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=_base["Heading1"], fontSize=17, leading=21, spaceAfter=4, textColor=colors.HexColor("#17202A"))
H2 = ParagraphStyle("H2", parent=_base["Heading2"], fontSize=12.5, leading=16, spaceBefore=10, spaceAfter=5)
H3 = ParagraphStyle("H3", parent=_base["Heading3"], fontSize=10, leading=13, spaceBefore=7, spaceAfter=3)
BODY = ParagraphStyle("Body", parent=_base["BodyText"], fontSize=8.5, leading=11)
SMALL = ParagraphStyle("Small", parent=BODY, fontSize=7.3, leading=9.2)
MUTED = ParagraphStyle("Muted", parent=SMALL, textColor=GREY)
CELL = ParagraphStyle("Cell", parent=BODY, fontSize=7.4, leading=9.2)
CELL_B = ParagraphStyle("CellB", parent=CELL, fontName="Helvetica-Bold")


def _p(text: object, style: ParagraphStyle = CELL) -> Paragraph:
    return Paragraph(escape(str(text)).replace("\n", "<br/>"), style)


def _zone(x: float, y: float, size: tuple[float, float]) -> str:
    col = min(7, max(0, int(8 * x / max(size[0], 1))))
    row = min(5, max(0, int(6 * y / max(size[1], 1))))
    return f"{'ABCDEFGH'[col]}{row + 1}"


def grid_place(el: Element | None) -> str | None:
    """Grid location of an element as the engineer looks for it ("B-12"), when it stands on a plan view.
    A shop element named by its own label is found by that label instead."""
    if el is None or not el.grid_ref or (el.source != "plan" and el.labeled):
        return None
    return el.grid_ref


def _where(el: Element | None, with_file: bool = False) -> str:
    """Human-readable location: grid axes, page, x, y (PDF points, top-left origin) and sheet zone."""
    if el is None:
        return "-"
    head = f"{el.fichier}\n" if with_file else f"{el.feuillet}, "
    place = grid_place(el)
    axes = f"axes {place}, " if place else ""
    return f"{head}{axes}p. {el.page}, x={el.x:.0f}, y={el.y:.0f} (zone {_zone(el.x, el.y, el.page_size)})"


def _bars(el: Element | None, limit: int = 4) -> str:
    if el is None:
        return "-"
    parts = [b.describe() for b in el.bars[:limit]]
    if len(el.bars) > limit:
        parts.append(f"+{len(el.bars) - limit}")
    return ", ".join(parts)


def _levels(el: Element | None) -> str:
    return f" (niv. {'@'.join(el.levels)})" if el is not None and el.levels else ""


def _table(rows: list[list], widths: list[float], header_bg=LIGHT, extra: list | None = None) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), header_bg),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, RULE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.25, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]
    table.setStyle(TableStyle(style + (extra or [])))
    return table


def _kpis(results: list[Result]) -> Table:
    counts = {s: sum(r.statut == s for r in results) for s in STATUS_LABEL}
    review = sum(r.a_valider for r in results)
    cells, style = [], []
    items = [(STATUS_LABEL[s], counts[s], STATUS_COLOR[s]) for s in (NON_CONFORME, MANQUANT, AJOUTE, CONFORME)]
    items.append(("À valider (confiance < 0,60)", review, GREY))
    for i, (label, value, color) in enumerate(items):
        cells.append(Paragraph(f'<font size="17" color="{color.hexval()}"><b>{value}</b></font><br/>'
                               f'<font size="7.5" color="#5F6B7A">{escape(label)}</font>', BODY))
        style.append(("LINEABOVE", (i, 0), (i, 0), 2.2, color))
    table = Table([cells], colWidths=[37 * mm] * len(items))
    table.setStyle(TableStyle(style + [("BACKGROUND", (0, 0), (-1, -1), LIGHT), ("TOPPADDING", (0, 0), (-1, -1), 6),
                                       ("BOTTOMPADDING", (0, 0), (-1, -1), 7), ("LEFTPADDING", (0, 0), (-1, -1), 7)]))
    return table


def _summary_table(results: list[Result]) -> Table:
    types: dict[str, str] = {}
    for r in results:
        types.setdefault(r.feuillet, r.type_element)
    head = ["Feuillet", "Type d'élément", "Conformes", "Non conformes", "Manquants", "Ajoutés", "À valider"]
    rows: list[list] = [[_p(h, CELL_B) for h in head]]
    totals = [0] * 5
    extra = []
    for i, (feuillet, c) in enumerate(summarize(results).items(), start=1):
        values = [c[CONFORME], c[NON_CONFORME], c[MANQUANT], c[AJOUTE], c["a_valider"]]
        totals = [a + b for a, b in zip(totals, values)]
        rows.append([_p(feuillet, CELL_B), _p(types.get(feuillet, ""))] + [_p(v) for v in values])
        if c[NON_CONFORME]:
            extra.append(("TEXTCOLOR", (3, i), (3, i), RED))
    rows.append([_p("Total", CELL_B), _p("")] + [_p(v, CELL_B) for v in totals])
    extra.append(("LINEABOVE", (0, len(rows) - 1), (-1, len(rows) - 1), 0.8, RULE))
    return _table(rows, [30 * mm, 38 * mm, 23 * mm, 27 * mm, 23 * mm, 21 * mm, 21 * mm], extra=extra)


def _finding_card(r: Result, cropper: Cropper | None) -> list:
    color = GRAVITE_COLOR.get(r.gravite, GREY)
    place = grid_place(r.plan)
    axes = f" - axes {place}" if place and place != r.element else ""
    title = (f'<font color="{RED.hexval()}"><b>{r.id}</b></font> &nbsp; <b>{escape(r.type_element.capitalize())} '
             f'{escape(r.element)}{escape(axes)}</b>{escape(_levels(r.plan or r.atelier))} &nbsp; '
             f'<font color="{color.hexval()}">gravité {r.gravite}</font> &nbsp; '
             f'<font color="#5F6B7A">confiance {r.confiance:.2f}{" - à valider" if r.a_valider else ""}</font>')
    flow: list = [Paragraph(title, BODY)]
    for e in r.ecarts:
        flow.append(Paragraph(f"&bull; {escape(e.message)}", BODY))
    if r.note:
        flow.append(Paragraph(escape(r.note), MUTED))
    where = [[_p("Plan : " + _where(r.plan), SMALL), _p("Atelier : " + _where(r.atelier, with_file=True), SMALL)]]
    flow.append(Table(where, colWidths=[91 * mm, 91 * mm], style=[("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                                    ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    if cropper is not None:
        images = []
        for el, boxes in ((r.plan, [e.plan_bbox for e in r.ecarts if e.plan_bbox]),
                          (r.atelier, [e.atelier_bbox for e in r.ecarts if e.atelier_bbox])):
            if el is None or not el.path:
                images.append("")
                continue
            marks = boxes or [(el.x - 14, el.y - 8, el.x + 14, el.y + 8)]
            try:
                crop = Image(io.BytesIO(cropper.crop(el.path, el.page, marks)), width=62 * mm, height=39 * mm,
                             kind="proportional")
                mini = Image(io.BytesIO(cropper.minimap(el.path, el.page, (el.x, el.y))), width=27 * mm,
                             height=39 * mm, kind="proportional")
                images.append(Table([[crop, mini]], colWidths=[63 * mm, 28 * mm],
                                    style=[("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
            except Exception:  # a crop is a convenience; never lose the report over one
                images.append("")
        flow.append(Table([images], colWidths=[91 * mm, 91 * mm], style=[("VALIGN", (0, 0), (-1, -1), "TOP"),
                                                                          ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
    flow.append(Spacer(1, 5))
    return [KeepTogether(flow)]


def _short_ecart(r: Result) -> str:
    """The discrepancies of a finding in a few words: "Quantité 6 -> 4 (25M)"."""
    if r.statut == MANQUANT:
        return "Absent à l'atelier : " + _bars(r.plan, 3)
    parts = []
    for e in r.ecarts:
        if e.attribut == "absence":
            parts.append(f"Introuvable : {e.plan}")
        elif e.attribut == "quantite":
            size = e.plan_bar.diametre if e.plan_bar is not None and e.plan_bar.diametre else ""
            parts.append(f"Quantité {e.plan} -> {e.atelier}" + (f" ({size})" if size else ""))
        elif e.attribut == "espacement_mm":
            parts.append(f"Espacement {e.plan} -> {e.atelier} mm")
        elif e.attribut == "longueur_mm":
            parts.append(f"Longueur {e.plan} -> {e.atelier} mm")
        else:
            parts.append(f"Diamètre {e.plan} -> {e.atelier}")
    return " ; ".join(parts)


def _findings_index(results: list[Result], limit: int) -> list:
    """Every non-conformity and missing element on one list, by sheet and grid place: the list an
    engineer (or a checker holding a list of known discrepancies) goes through first."""
    rows_in = [r for r in results if r.statut in (NON_CONFORME, MANQUANT)]
    rows_in.sort(key=lambda r: (r.feuillet, r.statut != NON_CONFORME, grid_place(r.plan) or "~", r.element))
    head = ["ID", "Feuillet", "Axes", "Élément", "Écart (plan -> atelier)", "Gravité", "Conf."]
    rows: list[list] = [[_p(h, CELL_B) for h in head]]
    extra = []
    for i, r in enumerate(rows_in[:limit], start=1):
        rows.append([_p(r.id), _p(r.feuillet), _p(grid_place(r.plan) or "-"), _p(f"{r.element}{_levels(r.plan)}"),
                     _p(_short_ecart(r)), _p(r.gravite if r.statut == NON_CONFORME else "majeur"),
                     _p(f"{r.confiance:.2f}{' *' if r.a_valider else ''}")])
        if r.statut == NON_CONFORME:
            extra.append(("TEXTCOLOR", (0, i), (0, i), RED))
    flow: list = [_table(rows, [17 * mm, 17 * mm, 22 * mm, 30 * mm, 66 * mm, 15 * mm, 13 * mm], extra=extra)]
    if len(rows_in) > limit:
        flow.append(_p(f"... et {len(rows_in) - limit} autres, listés dans le fichier JSON de comparaison.", MUTED))
    flow.append(_p("* à valider (confiance inférieure à 0,60). Les fiches détaillées, avec extraits d'image, suivent "
                   "par feuillet.", MUTED))
    return flow


def _list_table(results: list[Result], side: str, limit: int) -> list:
    head = ["ID", "Élément", "Armature", "Emplacement", "Conf."]
    rows: list[list] = [[_p(h, CELL_B) for h in head]]
    for r in results[:limit]:
        el = r.plan if side == "plan" else r.atelier
        rows.append([_p(r.id), _p(f"{r.element}{_levels(el)}"), _p(_bars(el)), _p(_where(el, with_file=side != "plan")),
                     _p(f"{r.confiance:.2f}")])
    flow: list = [_table(rows, [17 * mm, 34 * mm, 52 * mm, 67 * mm, 13 * mm])]
    if len(results) > limit:
        flow.append(_p(f"... et {len(results) - limit} autres, listés dans le fichier JSON de comparaison.", MUTED))
    return flow


def build_report(project: str, results: list[Result], coverage: Coverage, stats: dict, out_path: Path,
                 cfg: Config) -> Path:
    out_path = Path(out_path)
    doc = SimpleDocTemplate(str(out_path), pagesize=letter, leftMargin=15 * mm, rightMargin=15 * mm,
                            topMargin=15 * mm, bottomMargin=15 * mm, title=f"Rapport de conformité - {project}",
                            author="l2c-rebar")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(GREY)
        canvas.drawString(15 * mm, 9 * mm, f"{project} - vérification d'armature plan / atelier - confidentiel")
        canvas.drawRightString(letter[0] - 15 * mm, 9 * mm, f"page {canvas.getPageNumber()}")
        canvas.restoreState()

    story: list = [
        Paragraph(f"Rapport de conformité de l'armature - {escape(project)}", H1),
        Paragraph(f"Plans de structure comparés aux dessins d'atelier. Généré le {now} par l2c-rebar {__version__}. "
                  "La décision finale appartient à l'ingénieur.", MUTED),
        Spacer(1, 8),
        _kpis(results),
        Spacer(1, 6),
        Paragraph(
            f"{stats.get('plan_pages', 0)} pages de plan et {stats.get('atelier_pages', 0)} pages d'atelier lues "
            f"({stats.get('atelier_pages_ocr', 0)} par OCR local). {stats.get('plan_elements', 0)} éléments extraits "
            f"du plan, {stats.get('atelier_elements', 0)} de l'atelier. Les coordonnées x, y sont en points PDF depuis "
            "le coin supérieur gauche de la page ; la zone (A1 à H6) découpe le feuillet en 8 colonnes et 6 rangées.",
            SMALL),
        Paragraph("Résumé par feuillet de plan", H2),
        _summary_table(results),
    ]

    notes = []
    if coverage.types_sans_atelier:
        notes.append("Aucun dessin d'atelier fourni pour : " + ", ".join(coverage.types_sans_atelier) + ".")
    if coverage.feuillets_sans_atelier:
        shown = ", ".join(coverage.feuillets_sans_atelier[:25])
        more = len(coverage.feuillets_sans_atelier) - 25
        notes.append(f"Feuillets non couverts, en tout ou en partie, par les dessins d'atelier reçus : {shown}"
                     + (f" et {more} autres." if more > 0 else "."))
    if coverage.feuillets_lecture_partielle:
        items = ", ".join(f"{sheet} ({ratio})" for sheet, ratio in sorted(coverage.feuillets_lecture_partielle.items()))
        notes.append("Feuillets dont moins de la moitié des éléments ont été retrouvés à l'atelier (dessins non fournis "
                     f"ou lecture incomplète) ; les éléments non retrouvés n'y sont pas listés un à un : {items}.")
    if coverage.reperes_non_fiables:
        notes.append("Appariement par repère jugé non fiable (presque aucun accord) et remplacé par une comparaison "
                     "par contenu pour : " + ", ".join(coverage.reperes_non_fiables) + ". À vérifier manuellement.")
    if coverage.ocr_lignes_recollees:
        notes.append("Dessins d'atelier lus par OCR : les morceaux d'une même ligne de texte, lus séparément, ont été "
                     "recollés pour : " + ", ".join(coverage.ocr_lignes_recollees) + " (lecture qui confirme le plus "
                     "souvent le plan).")
    if coverage.accord_par_type:
        notes.append("Accord plan / atelier des éléments appariés par repère (part des barres confirmées) : "
                      + ", ".join(f"{t} {round(100 * a)} %" for t, a in sorted(coverage.accord_par_type.items()))
                      + ". Sous 90 %, les écarts de ce type voient leur confiance réduite d'autant.")
    if coverage.non_verifies:
        notes.append(f"{coverage.non_verifies} annotations de dalle écrites sans diamètre (par exemple « 12(6) ») n'ont pu être "
                     "ni confirmées ni contredites : elles sont dans le JSON, pas dans les écarts.")
    if coverage.atelier_non_apparies:
        notes.append(f"{coverage.atelier_non_apparies} annotations d'atelier sans repère n'ont pas d'équivalent direct "
                     "au plan (détail d'atelier plus fin que le plan) ; elles ne sont pas comptées comme ajouts.")
    if stats.get("atelier_pages_vides"):
        notes.append(f"{stats['atelier_pages_vides']} pages d'atelier sans texte lisible (OCR désactivé ou vide).")
    if notes:
        story.append(Paragraph("Couverture de la vérification", H2))
        story += [Paragraph(f"&bull; {escape(n)}", BODY) for n in notes]

    if any(r.statut in (NON_CONFORME, MANQUANT) for r in results):
        story.append(PageBreak())
        story.append(Paragraph("Liste des écarts, par feuillet et par axes", H2))
        story += _findings_index(results, cfg.max_index_rows)

    by_sheet: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_sheet[r.feuillet].append(r)

    cropper = Cropper() if cfg.crops else None
    cards_left = cfg.max_cards
    try:
        for feuillet in sorted(by_sheet):
            rows = by_sheet[feuillet]
            nc = sorted((r for r in rows if r.statut == NON_CONFORME),
                        key=lambda r: ({"critique": 0, "majeur": 1, "mineur": 2}[r.gravite], -r.confiance))
            missing = [r for r in rows if r.statut == MANQUANT]
            added = [r for r in rows if r.statut == AJOUTE]
            if not (nc or missing or added):
                continue
            ok = sum(r.statut == CONFORME for r in rows)
            story.append(PageBreak())
            story.append(Paragraph(f"Feuillet {escape(feuillet)} - {escape(rows[0].type_element)}", H2))
            story.append(Paragraph(f"{ok} conformes, {len(nc)} non conformes, {len(missing)} manquants à l'atelier, "
                                   f"{len(added)} ajoutés à l'atelier.", BODY))
            if nc:
                story.append(Paragraph("Non-conformités", H3))
                for r in nc:
                    story += _finding_card(r, cropper if cards_left > 0 else None)
                    cards_left -= 1
            if missing:
                story.append(Paragraph("Présents au plan, introuvables dans les dessins d'atelier", H3))
                story += _list_table(missing, "plan", cfg.max_detail_rows)
            if added:
                story.append(Paragraph("Présents à l'atelier, sans équivalent au plan", H3))
                story += _list_table(added, "atelier", cfg.max_detail_rows)

        story.append(PageBreak())
        story.append(Paragraph("Méthode et limites", H2))
        for line in (
            "Texte lu dans la couche vectorielle des PDF ; les pages sans texte sont lues par OCR local (aucun envoi externe).",
            "Appariement par repère d'élément (C-12, S3...) quand il existe des deux côtés, sinon par contenu "
            "(diamètre, quantité, espacement) et position, à l'intérieur du même type d'élément et du même niveau.",
            "Seuls les attributs présents des deux côtés sont comparés. Une quantité au plan peut être répartie sur "
            "plusieurs repères à l'atelier : la somme est alors comparée.",
            "Gravité : critique = moins d'acier que le plan (quantité ou diamètre inférieur, espacement supérieur) ; "
            "majeur = barres introuvables, diamètre supérieur ou longueur plus courte ; mineur = surplus.",
            "Confiance < 0,60 : résultat à valider par l'ingénieur (texte OCR, appariement par contenu, niveau incertain).",
        ):
            story.append(Paragraph(f"&bull; {escape(line)}", BODY))
        doc.build(story, onFirstPage=footer, onLaterPages=footer)
    finally:
        if cropper is not None:
            cropper.close()
    return out_path
