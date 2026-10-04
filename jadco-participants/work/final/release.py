"""Validate, document and package the JADCO submission. Run with --rebuild to execute first."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as element_tree
import zipfile
from datetime import datetime, timezone
from pathlib import Path


from lint_notebook import lint

HERE = Path(__file__).resolve().parent
PRESENTATION = HERE.parent / "presentation"
DEFAULT_ARCHIVE = HERE.parent / "submission" / "jadco_submission.zip"
TOOLS = ("build_notebook.py", "lint_notebook.py", "refresh_docs.py", "release.py", "show_outputs.py")
OUTPUTS = ("final_answer.json", "evidence.json", "aggregates_2026.csv", "model_2026.json", "validation.json")
OPTIONAL_OUTPUTS = ("dashboard_2026.html", "reproducibility.json")
PRESENTATION_FILES = ("presentation.pdf", "PRESENTATION.md", "make_presentation.py", "render_pdf.py")


def run_script(*arguments, **kwargs):
    """Use the invoking interpreter so validation and execution share an environment."""
    subprocess.run([sys.executable, *map(str, arguments)], cwd=HERE, check=True, **kwargs)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dependency_versions():
    names = [line.split("==", 1)[0] for line in (HERE / "requirements.txt").read_text().splitlines()
             if line.strip() and not line.startswith("#")]
    return {name: importlib.metadata.version(name) for name in names}


def validate_delivery():
    notebook_path = HERE / "equinoxe_hausse_2026.ipynb"
    findings, cell_count = lint(notebook_path)
    if findings:
        raise RuntimeError("Notebook gate failed:\n" + "\n".join(map(str, findings)))
    with tempfile.TemporaryDirectory(prefix="jadco-test-report-") as temporary:
        report_path = Path(temporary) / "pytest.xml"
        run_script("-m", "pytest", "tests", "-q", f"--junitxml={report_path}")
        suites = element_tree.parse(report_path).getroot()
        tests_passed = sum(int(suite.attrib["tests"]) for suite in suites.iter("testsuite"))
    record = {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "dependencies": dependency_versions(),
        "notebook_cells": cell_count, "lint_findings": len(findings), "tests_passed": tests_passed,
        "numerical_artifacts_sha256": {name: sha256(HERE / "outputs" / name) for name in OUTPUTS if name != "validation.json"},
        "notebook_sha256": sha256(notebook_path),
        "sources_sha256": {path.name: sha256(path) for path in sorted(HERE.glob("nb_part*.py"))},
    }
    reproduction_path = HERE / "outputs/reproducibility.json"
    if reproduction_path.exists():
        reproduction = json.loads(reproduction_path.read_text(encoding="utf-8"))
        for field in ("numerical_artifacts_sha256", "sources_sha256"):
            if reproduction[field] != record[field]:
                raise RuntimeError(f"Stale independent reproduction record: {field}")
        if not all(reproduction["numerical_artifacts_reproduced"].values()):
            raise RuntimeError("Independent reconstruction did not reproduce all numerical artifacts")
        record["independent_reconstruction"] = reproduction["numerical_artifacts_reproduced"]
    pending = HERE / "outputs" / "validation.pending.json"
    pending.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    pending.replace(HERE / "outputs" / "validation.json")
    return record


def copy_file(source, staging, relative_path):
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"Expected a regular deliverable file: {source}")
    target = staging / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def package_delivery(archive_path, validation):
    """Package an explicit file list, excluding CRM extracts, exploratory work and caches."""
    archive_path = Path(archive_path).resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=archive_path.parent) as temporary:
        staging = Path(temporary) / "delivery"
        staging.mkdir()
        files = [HERE / name for name in (*TOOLS, "README.md", "AUDIT_PROGRESS.md", "requirements.txt", "equinoxe_hausse_2026.ipynb")]
        files += sorted(HERE.glob("nb_part*.py"))
        files += sorted((HERE / "tests").glob("test_*.py"))
        files += [HERE / "external" / name for name in ("SOURCES.md", "fetch.py", "manual_public_rates.csv")]
        files += [HERE / "external/tidy" / name for name in ("cpi_rent_monthly.csv", "external_annual.csv")]
        files += [HERE / "outputs" / name for name in OUTPUTS]
        files += [HERE / "outputs" / name for name in OPTIONAL_OUTPUTS if (HERE / "outputs" / name).is_file()]
        for source in files:
            copy_file(source, staging, Path("final") / source.relative_to(HERE))
        for name in PRESENTATION_FILES:
            copy_file(PRESENTATION / name, staging, Path("presentation") / name)
        answer = json.loads((HERE / "outputs/final_answer.json").read_text(encoding="utf-8"))
        (staging / "README.md").write_text(
            "# Collection Équinoxe — livraison JADCO\n\n"
            f"Hausse effective : {answer['headline_effective_pct']:.2f} %. "
            f"Hausse contractuelle : {answer['contract_pct']:.2f} %.\n\n"
            "Ouvrir `final/equinoxe_hausse_2026.ipynb` et `presentation/presentation.pdf`.\n"
            "La définition, les scénarios et les limites figurent dans `final/README.md`.\n\n"
            "Pour reconstruire : installer `final/requirements.txt`, puis depuis `final/` :\n\n"
            "```bash\nexport JADCO_DATA_DIR=/chemin/vers/les/quatre/csv\npython release.py --rebuild\n```\n\n"
            "Les extractions CRM ne sont pas incluses. Les tables publiques sont livrées ; aucun téléchargement n'est nécessaire.\n"
            f"Validation : {validation['tests_passed']} tests passent, {validation['lint_findings']} finding de lint.\n"
            "`MANIFEST.json` contient les empreintes SHA-256 de tous les fichiers du bundle.\n",
            encoding="utf-8",
        )
        payload = {str(path.relative_to(staging)): sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {"files_sha256": payload, "validation": validation}
        (staging / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        pending = Path(temporary) / "submission.zip"
        with zipfile.ZipFile(pending, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, str(path.relative_to(staging)))
        with zipfile.ZipFile(pending) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("Corrupt submission archive")
            for name, expected in payload.items():
                actual = hashlib.sha256(archive.read(name)).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Checksum mismatch: {name}")
        pending.replace(archive_path)
    return archive_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="Execute from the percent-format sources before validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARCHIVE)
    arguments = parser.parse_args()
    if arguments.rebuild:
        run_script("build_notebook.py", "--execute")
    validation = validate_delivery()
    run_script("refresh_docs.py")
    environment = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix="jadco-plot-cache-") as cache:
        environment["MPLCONFIGDIR"] = cache
        run_script(PRESENTATION / "render_pdf.py", env=environment)
    archive_path = package_delivery(arguments.output, validation)
    print(f"Verified submission: {archive_path} ({archive_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
