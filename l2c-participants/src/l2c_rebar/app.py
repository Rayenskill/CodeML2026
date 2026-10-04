"""Streamlit interface: run a project, browse the findings, validate the uncertain ones.

    python -m l2c_rebar ui

Everything runs on this machine; the app makes no network call.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from l2c_rebar.config import Config
from l2c_rebar.models import AJOUTE, CONFORME, MANQUANT, NON_CONFORME
from l2c_rebar.pipeline import run_project
from l2c_rebar.report.annotate import annotate_project
from l2c_rebar.report.crops import Cropper
from l2c_rebar.report.pdf_report import build_report
from l2c_rebar.validation import CONFIRMED, REJECTED, apply_validations, load_validations, result_key, save_validations

LABELS = {NON_CONFORME: "Non conforme", MANQUANT: "Manquant à l'atelier", AJOUTE: "Ajouté à l'atelier",
          CONFORME: "Conforme"}

st.set_page_config(page_title="L2C - Armature : plan vs atelier", layout="wide")
st.title("Vérification d'armature : plans vs dessins d'atelier")

with st.sidebar:
    project_dir = st.text_input("Dossier du projet", placeholder=r"C:\...\MonProjet")
    out_root = st.text_input("Dossier de sortie", value="outputs")
    ocr = st.selectbox("OCR local des pages sans texte", ["auto", "off"], help="auto : seulement les pages sans texte")
    crops = st.checkbox("Extraits d'image dans le rapport", value=True)
    launch = st.button("Analyser", type="primary", disabled=not project_dir)

if launch:
    project = Path(project_dir)
    out_dir = Path(out_root) / project.name
    status = st.status("Analyse en cours...", expanded=True)
    out = run_project(project, out_dir, Config(ocr=ocr, crops=crops), progress=status.write)
    status.update(label=f"Analyse terminée en {out.stats['duree_s']} s", state="complete")
    st.session_state["run"] = out

out = st.session_state.get("run")
if out is None:
    st.info("Indiquez le dossier d'un projet (plan PDF à la racine, dessins d'atelier dans les sous-dossiers), puis Analyser.")
    st.stop()

validation_path = out.out_dir / "validations.json"
validations = load_validations(validation_path)
results = out.results

cols = st.columns(5)
for col, statut in zip(cols, (NON_CONFORME, MANQUANT, AJOUTE, CONFORME)):
    col.metric(LABELS[statut], sum(r.statut == statut for r in results))
cols[4].metric("À valider", sum(r.a_valider and result_key(r) not in validations for r in results))

rows = [{"ID": r.id, "Feuillet": r.feuillet, "Type": r.type_element, "Élément": r.element, "Statut": LABELS[r.statut],
         "Gravité": r.gravite if r.ecarts else "", "Confiance": r.confiance,
         "Écart": " ; ".join(e.message for e in r.ecarts),
         "Verdict": validations.get(result_key(r), {}).get("verdict", "")} for r in results]
table = pd.DataFrame(rows)

left, mid, right = st.columns(3)
statuses = left.multiselect("Statut", list(LABELS.values()), default=[LABELS[NON_CONFORME], LABELS[MANQUANT]])
sheets = mid.multiselect("Feuillet", sorted(table["Feuillet"].unique()))
only_uncertain = right.checkbox("Seulement les cas à valider (confiance < 0,60)")
view = table[table["Statut"].isin(statuses)] if statuses else table
if sheets:
    view = view[view["Feuillet"].isin(sheets)]
if only_uncertain:
    view = view[view["Confiance"] < 0.6]
st.dataframe(view, use_container_width=True, hide_index=True, height=320)

st.subheader("Examiner un résultat")
choice = st.selectbox("Résultat", view["ID"].tolist())
selected = next((r for r in results if r.id == choice), None)
if selected is not None:
    st.markdown(f"**{selected.id} - {selected.type_element} {selected.element}** - confiance {selected.confiance:.2f}")
    for e in selected.ecarts:
        st.markdown(f"- {e.message} *(gravité {e.gravite})*")
    if selected.note:
        st.caption(selected.note)
    cropper = Cropper()
    a, b = st.columns(2)
    for col, el, boxes, name in ((a, selected.plan, [e.plan_bbox for e in selected.ecarts if e.plan_bbox], "Plan"),
                                 (b, selected.atelier, [e.atelier_bbox for e in selected.ecarts if e.atelier_bbox],
                                  "Atelier")):
        if el is None:
            col.write(f"{name} : aucun équivalent")
            continue
        col.caption(f"{name} : {el.fichier}, feuillet {el.feuillet}, p. {el.page}, x={el.x:.0f}, y={el.y:.0f}")
        marks = boxes or [(el.x - 14, el.y - 8, el.x + 14, el.y + 8)]
        col.image(cropper.crop(el.path, el.page, marks, width=420, height=260))
    cropper.close()

    key = result_key(selected)
    current = validations.get(key, {})
    options = ["", CONFIRMED, REJECTED]
    verdict = st.radio("Verdict de l'ingénieur", options, index=options.index(current.get("verdict", "")),
                       format_func=lambda v: {"": "Non examiné", CONFIRMED: "Confirmé", REJECTED: "Rejeté (fausse alerte)"}[v],
                       horizontal=True)
    comment = st.text_input("Commentaire", value=current.get("commentaire", ""))
    if st.button("Enregistrer le verdict"):
        if verdict:
            validations[key] = {"verdict": verdict, "commentaire": comment, "id": selected.id}
        else:
            validations.pop(key, None)
        save_validations(validation_path, validations)
        st.success(f"Enregistré dans {validation_path}")

st.subheader("Livrables")
if st.button("Générer le rapport PDF et les PDF annotés (verdicts appliqués)"):
    final = apply_validations(list(results), validations)
    report = build_report(out.project, final, out.coverage, out.stats, out.out_dir / f"{out.project}_rapport.pdf",
                          Config(crops=crops))
    annotate_project(final, out.out_dir)
    st.session_state["report"] = report
report = st.session_state.get("report")
if report and Path(report).exists():
    st.download_button("Télécharger le rapport PDF", Path(report).read_bytes(), file_name=Path(report).name)
for name, path in out.files.items():
    st.download_button(f"Télécharger {path.name}", path.read_bytes(), file_name=path.name, key=name)
st.caption(f"PDF annotés : {out.out_dir / 'annotes'}")
