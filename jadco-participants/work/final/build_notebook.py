"""Assemble nb_part*.py (percent format) into equinoxe_hausse_2026.ipynb and optionally execute it.

Usage:  python build_notebook.py [--execute] [--parts 1 2 3]
The percent-format sources are the single source of truth; the notebook is a build artefact.
"""
import argparse
import hashlib
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / "equinoxe_hausse_2026.ipynb"
CELL_MARKER = re.compile(r"^# %%(?: \[markdown\])?\s*$", re.MULTILINE)


def parse_percent(text):
    """Yield (kind, source) tuples from a percent-format script."""
    marker_positions = [(m.start(), m.group(0)) for m in CELL_MARKER.finditer(text)]
    for index, (start, marker) in enumerate(marker_positions):
        end = marker_positions[index + 1][0] if index + 1 < len(marker_positions) else len(text)
        body = text[start + len(marker): end].strip("\n")
        if "[markdown]" in marker:
            lines = [re.sub(r"^# ?", "", line) for line in body.splitlines()]
            yield "markdown", "\n".join(lines).strip()
        else:
            yield "code", body.strip()


def build(parts):
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = []
    for part in parts:
        text = (HERE / f"nb_part{part:02d}.py").read_text(encoding="utf-8")
        for kind, source in parse_percent(text):
            if not source:
                continue
            cell = nbformat.v4.new_markdown_cell(source) if kind == "markdown" else nbformat.v4.new_code_cell(source)
            cell.id = hashlib.sha256(f"{part}:{len(cells)}:{source}".encode()).hexdigest()[:16]
            cells.append(cell)
    notebook.cells = cells
    return notebook


def sync_public_data():
    """Copy the tidy PUBLIC external tables next to the notebook (never the CRM extract)."""
    source = HERE.parent / "external"
    target = HERE / "external"
    if source.is_dir():
        (target / "tidy").mkdir(parents=True, exist_ok=True)
        for csv_file in (source / "tidy").glob("*.csv"):
            shutil.copy2(csv_file, target / "tidy" / csv_file.name)
        for name in ("SOURCES.md", "fetch.py", "manual_public_rates.csv"):
            shutil.copy2(source / name, target / name)
    required = ("cpi_rent_monthly.csv", "external_annual.csv")
    missing = [name for name in required if not (target / "tidy" / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing bundled public tables: {missing}")


def execute_and_publish(notebook, output_path, publish_outputs=True):
    """Execute into a temporary output directory; publish only after a successful run.

    A failed kernel cannot leave the answer JSON or CSVs newer than the shipped notebook.
    The local kernel inherits the caller's Python environment, with isolated writable caches.
    """
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".notebook-build-", dir=HERE) as temporary:
        staging = Path(temporary)
        staged_outputs = staging / "outputs"
        environment = os.environ.copy()
        environment.pop("MPLBACKEND", None)
        environment["JADCO_OUTPUT_DIR"] = str(staged_outputs)
        environment["MPLCONFIGDIR"] = str(staging / "matplotlib")
        client = NotebookClient(notebook, timeout=1800, kernel_name="python3",
                                resources={"metadata": {"path": str(HERE)}}, allow_errors=False)
        try:
            client.execute(env=environment)
        except Exception:
            failed_path = output_path.with_suffix(".failed.ipynb")
            nbformat.write(notebook, failed_path)
            print(f"Partial execution saved to {failed_path}", file=sys.stderr)
            raise
        staged_notebook = output_path.with_suffix(".pending.ipynb")
        nbformat.write(notebook, staged_notebook)
        old_outputs = staging / "previous_outputs"
        output_directory = HERE / "outputs"
        had_previous_outputs = output_directory.exists()
        outputs_backed_up = False
        outputs_promoted = False
        try:
            if publish_outputs:
                if had_previous_outputs:
                    output_directory.rename(old_outputs)
                    outputs_backed_up = True
                staged_outputs.rename(output_directory)
                outputs_promoted = True
            staged_notebook.replace(output_path)
        except Exception:
            if publish_outputs:
                if outputs_promoted:
                    shutil.rmtree(output_directory)
                if outputs_backed_up:
                    old_outputs.rename(output_directory)
            staged_notebook.unlink(missing_ok=True)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parts", type=int, nargs="*")
    parser.add_argument("--out", default=str(NOTEBOOK))
    args = parser.parse_args()
    if args.parts and Path(args.out).resolve() == NOTEBOOK.resolve():
        parser.error("Partial builds require --out so the delivered notebook stays complete")
    sync_public_data()
    parts = args.parts or sorted(int(re.search(r"(\d+)", p.stem).group(1)) for p in HERE.glob("nb_part*.py"))
    notebook = build(parts)
    if args.execute:
        try:
            execute_and_publish(notebook, args.out, publish_outputs=not args.parts)
        except Exception as error:
            print("EXECUTION FAILED:", str(error)[-3000:])
            sys.exit(1)
    else:
        nbformat.write(notebook, args.out)
    print(f"wrote {args.out}: {len(notebook.cells)} cells ({sum(c.cell_type == 'code' for c in notebook.cells)} code)")


if __name__ == "__main__":
    main()
