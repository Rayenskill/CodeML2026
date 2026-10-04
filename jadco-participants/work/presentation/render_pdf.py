"""Build the jury deck from executed aggregate evidence, without embedding CRM rows."""
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

from make_presentation import ANSWER_PATH, build, fr

HERE = Path(__file__).resolve().parent
BACKGROUND = "#f5f7fa"
INK = "#152638"
MUTED = "#59697b"
TEAL = "#087f75"
BLUE = "#346caa"
ORANGE = "#bf642d"
TOTAL_SLIDES = 8


def base_slide(title, subtitle, number, answer):
    figure = plt.figure(figsize=(16, 9), facecolor=BACKGROUND)
    figure.text(.055, .94, "COLLECTION ÉQUINOXE  /  CODEML", color=TEAL, fontsize=15, weight="bold")
    figure.text(.055, .845, title, color=INK, fontsize=31, weight="bold")
    figure.text(.055, .785, subtitle, color=MUTED, fontsize=17)
    figure.text(.055, .035, f"Source : notebook exécuté · données au {answer['data_as_of']} · agrégats uniquement", fontsize=11, color=MUTED)
    figure.text(.94, .035, f"{number} / {TOTAL_SLIDES}", fontsize=12, color=TEAL, ha="right")
    return figure


def text_block(figure, text, x=.63, y=.68, width=36, size=21):
    for paragraph in text.split("\n"):
        lines = textwrap.wrap(paragraph, width=width) or [""]
        for line in lines:
            figure.text(x, y, line, fontsize=size, color=INK)
            y -= .045
        y -= .022
    if y < .095:
        raise ValueError("Slide text overflows; shorten the narrative")


def chart_axes(figure, position=(.075, .18, .49, .52)):
    axes = figure.add_axes(position, facecolor=BACKGROUND)
    axes.spines[["top", "right"]].set_visible(False)
    axes.tick_params(labelsize=13, colors=MUTED)
    axes.grid(axis="y", alpha=.15)
    axes.set_axisbelow(True)
    return axes


def label_bars(axes, bars, values):
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .1, f"{fr(value, 2)} %", ha="center", color=INK, fontsize=18, weight="bold")


def cards(figure, items, y=.34, height=.34):
    for index, (heading, body, color) in enumerate(items):
        x = .075 + index * .44
        box = FancyBboxPatch((x, y), .40, height, boxstyle="round,pad=0.015", facecolor="white", edgecolor=color, linewidth=2, transform=figure.transFigure)
        figure.add_artist(box)
        figure.text(x + .02, y + height - .06, heading, fontsize=23, weight="bold", color=color)
        text_block(figure, body, x=x + .02, y=y + height - .13, width=31, size=18)


def draw_slides(answer, evidence):
    diagnostics = answer["presentation_diagnostics"]
    figure = base_slide("Le mix n’est pas une hausse de prix", f"Même année, trois mesures : {diagnostics['mix_year']}", 1, answer)
    axes = chart_axes(figure)
    values = [diagnostics[key] for key in ("naive_growth_pct", "fixed_panel_growth_pct", "fisher_growth_pct")]
    bars = axes.bar(["Tous les baux", "Panel fixe", "Fisher"], values, color=[ORANGE, BLUE, TEAL], width=.58)
    label_bars(axes, bars, values)
    axes.set_ylim(0, max(values) * 1.25)
    axes.set_ylabel("Variation du loyer (%)", fontsize=15)
    text_block(figure, "Des immeubles entrent dans le portefeuille.\nLe loyer médian monte même sans hausse comparable dans chaque appartement.\nNotre réponse compare chaque unité à elle-même.")
    yield figure

    figure = base_slide("Une définition que l’on peut défendre", "Même unité · deux loyers · croissance annualisée", 2, answer)
    cards(figure, [("Effectif — mesure principale", "Loyer net des concessions.\nMédiane du changement typique par appartement.", TEAL),
                   ("Contractuel — mesure parallèle", "Loyer écrit au bail.\nRenouvellements et relocations analysés séparément.", BLUE)])
    figure.text(.075, .18, "Pour le budget : la moyenne pondérée par le loyer est aussi publiée.", color=MUTED, fontsize=21)
    yield figure

    figure = base_slide("La bonne clé fait toute la différence", "Le code de propriété distingue les appartements qui partagent un numéro", 3, answer)
    cards(figure, [("Clé retenue", "prop_code + unit_code\nMême appartement, bail précédent → bail suivant.", TEAL),
                   ("Contre-exemple testé", "site + unit_code\nDes propriétés distinctes partagent des codes d’appartement.", ORANGE)])
    figure.text(.075, .18, "Entonnoir de validité, annualisation exacte et sensibilité aux filtres dans le notebook.", color=MUTED, fontsize=20)
    yield figure

    figure = base_slide("Les concessions changent le revenu", "Les deux mesures restent visibles ; leur écart n’est pas une erreur", 4, answer)
    axes = chart_axes(figure)
    history = evidence["historical_growth"]
    years = [row["year"] for row in history]
    axes.plot(years, [row["contract"] for row in history], "o-", color=BLUE, linewidth=2.5, label="Contractuel")
    axes.plot(years, [row["effective"] for row in history], "o-", color=TEAL, linewidth=2.5, label="Effectif")
    axes.set_ylabel("Médiane à unité constante (%)", fontsize=15)
    axes.legend(fontsize=14)
    text_block(figure, f"{fr(diagnostics['concession_penetration_pct'])} % des baux récents ont une concession.\n"
               f"Reconstruction à ≤ {fr(diagnostics['reconstruction_tolerance_dollars'])} $ : {fr(diagnostics['reconstruction_share_pct'])} % des baux.\n"
               "Modèle à deux marges : qui reçoit une concession, puis quelle profondeur.")
    yield figure

    figure = base_slide("Deux provinces, une hypothèse de politique", "Le calendrier du TAL est un ancrage de prévision, pas un plafond universel", 5, answer)
    cards(figure, [("Québec", "Avis datés dans leur fenêtre légale.\nRéforme et ancienne méthode distinguées.", BLUE),
                   ("The Met — Ottawa", "Exemption Ontario étudiée.\nPolitique commune supposée ; calendrier QC utilisé comme proxy.", TEAL)], height=.38, y=.30)
    figure.text(.075, .17, "Les dates d’avis réelles manquent : les deux bornes du calendrier sont testées.", color=MUTED, fontsize=20)
    yield figure

    figure = base_slide("L’information disponible à chaque date", "La même fonction prévoit le futur et rejoue le passé", 6, answer)
    steps = [("1 · Historique connu", "Baux et affichages\ntronqués avant l’année"),
             ("2 · Sources datées", "IPC, SCHL, TAL, Ontario\nfiltrés par publication"),
             ("3 · Cohorte simulée", "Renouvellement, relocation,\nconcession par propriété")]
    for index, (title, body) in enumerate(steps):
        x = .065 + index * .31
        box = FancyBboxPatch((x, .43), .27, .25, boxstyle="round,pad=0.01", facecolor="white", edgecolor=TEAL, transform=figure.transFigure)
        figure.add_artist(box)
        figure.text(x + .015, .62, title, fontsize=20, weight="bold", color=TEAL)
        figure.text(x + .015, .51, body, fontsize=17, color=INK, linespacing=1.6)
    figure.text(.075, .27, "Trois millésimes d’information sont comparés séparément dans le backtest.", fontsize=21, color=INK)
    figure.text(.075, .17, "Structure repérée sur l’historique : la validation reste favorable au modèle.", fontsize=20, color=MUTED)
    yield figure

    figure = base_slide("Mesurer l’erreur, montrer les limites", "Comparaison à des références simples sur les années exigées", 7, answer)
    axes = chart_axes(figure)
    required_years = set(evidence["backtest_years"])
    structural = [row for row in evidence["backtest"] if row["model"] == "structural" and row["vintage"] == evidence["live_vintage"] and row["year"] in required_years]
    names = ["Modèle", "Reconduire", "Moyenne", "Dérive"]
    values = [np.mean([abs(row["error_effective"]) for row in structural])]
    for name in ("reconduire", "moyenne récente", "dérive"):
        rows = [row for row in evidence["baselines"] if row["kind"] == "effective" and row["method"] == name and row["year"] in required_years]
        if len(rows) != len(required_years):
            raise ValueError(f"Incomplete baseline evidence: {name}")
        values.append(np.mean([abs(row["error"]) for row in rows]))
    bars = axes.bar(names, values, color=[TEAL, MUTED, MUTED, MUTED], width=.6)
    for bar, value in zip(bars, values):
        axes.text(bar.get_x() + bar.get_width() / 2, value + .035, fr(value, 2), ha="center", fontsize=18, weight="bold", color=INK)
    axes.set_ylim(0, max(values) * 1.25)
    axes.set_ylabel("Erreur absolue moyenne — points", fontsize=15)
    text_block(figure, f"Années : {', '.join(map(str, sorted(required_years)))}.\n"
               "Le régime ancien est moins bien prévu.\n"
               f"Biais récent effectif : {fr(answer['backtest_bias_points']['effective'], 2)} point.\n"
               "Ces années ont aussi servi à choisir la structure.")
    yield figure

    figure = base_slide("Notre réponse pour 2026", "Une estimation centrale, des scénarios de politique et une incertitude explicite", 8, answer)
    axes = chart_axes(figure, position=(.075, .26, .49, .39))
    estimates = [answer["headline_effective_pct"], answer["contract_pct"]]
    bands = [answer["headline_band_pct"], answer["contract_band_pct"]]
    for index, (estimate, band, color) in enumerate(zip(estimates, bands, [TEAL, BLUE])):
        axes.errorbar(estimate, 1 - index, xerr=[[estimate - band[0]], [band[1] - estimate]], fmt="o", markersize=12, capsize=9, linewidth=3, color=color)
        axes.text(estimate, 1 - index + .2, f"{fr(estimate, 2)} %", ha="center", fontsize=25, weight="bold", color=color)
    axes.set_yticks([1, 0], ["Effectif", "Contractuel"])
    axes.set_ylim(-.55, 1.55)
    axes.set_xlim(min(band[0] for band in bands) - .5, max(band[1] for band in bands) + .5)
    axes.set_xlabel("Croissance (%) · bande prédictive 10–90 %", fontsize=14)
    text_block(figure, f"{answer['cohort_units']} unités dans la cohorte.\n"
               f"Sensibilité au biais : {fr(answer['bias_adjusted_sensitivity_pct']['effective'], 2)} % ; correction non validée.\n"
               "Les décisions de politique dominent l’incertitude.\n"
               "Les poids uniformes restent un jugement.")
    figure.text(.075, .14, "Définition, appariement, concessions, sources et validation : un raisonnement reproductible.", fontsize=18, color=MUTED)
    yield figure


def main():
    answer = json.loads(ANSWER_PATH.read_text(encoding="utf-8"))
    evidence = json.loads((ANSWER_PATH.parent / "evidence.json").read_text(encoding="utf-8"))
    (HERE / "PRESENTATION.md").write_text(build(answer), encoding="utf-8")
    output = HERE / "presentation.pdf"
    pending = output.with_suffix(".pending.pdf")
    slides = list(draw_slides(answer, evidence))
    try:
        with PdfPages(pending, metadata={"Title": "Collection Équinoxe — hausse de loyer 2026", "Author": "Équipe CodeML"}) as pdf:
            for index, figure in enumerate(slides, 1):
                pdf.savefig(figure)
                if index in (1, 7, TOTAL_SLIDES):
                    figure.savefig(HERE / f"slide_{index:02d}.png", dpi=90)
        pending.replace(output)
    finally:
        pending.unlink(missing_ok=True)
        for figure in slides:
            plt.close(figure)
    print(f"{output}: {len(slides)} slides, computed charts")


if __name__ == "__main__":
    main()
