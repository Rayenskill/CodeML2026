"""Inspect stored notebook outputs without executing the CRM analysis again."""
import argparse
from pathlib import Path

import nbformat


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("first_cell", type=int, nargs="?", default=0)
    arguments = parser.parse_args()
    notebook = nbformat.read(arguments.notebook, as_version=4)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code" or index < arguments.first_cell:
            continue
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/plain") or ""
            if output.get("output_type") == "error":
                text = f"ERROR: {output.get('ename', '')} {output.get('evalue', '')}"
            if text:
                print(f"Cell {index}\n{text[:6000]}")


if __name__ == "__main__":
    main()
