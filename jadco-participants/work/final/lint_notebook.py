"""Notebook quality gate for the JADCO deliverable (run before every hand-in; enforced by tests/test_deliverable.py).

Checks (the standard a quant desk would apply, no exemptions):
  markdown       a Markdown cell precedes every data-modifying step (at most MAX_CODE_CELLS_AFTER_MARKDOWN code cells in between);
  magic-number   numeric literals only in `# CONFIG` cells; elsewhere only the identities {0, 1, 2, 100}
                 (no exemption for UPPER_CASE names, dataclass fields, Timestamp arguments, f-strings or plotting keywords);
  prose-number   no percentage, amount or decimal typed in a Markdown cell (numbers in the text are written by code);
                 inline code (`...`) and URLs are not prose;
  repeated-date  no hand-built date such as Timestamp(year - 1, 12, 31) (one named helper: year_start / year_end / vintage_cutoff);
  dead-name      no top-level name assigned and never read;
  label-spaces   no label told apart from another by leading or trailing spaces;
  wrong-key      never match units on site + unit code outside the cell that demonstrates the failure;
  naming         snake_case identifiers, no throw-away names (df2, tmp, x1 ...);
  crm-output     no unit identifiers in stored outputs or in the files written to outputs/;
  executed       every code cell executed top to bottom without error;
  structure      the expected number of numbered sections.

Usage: python lint_notebook.py equinoxe_hausse_2026.ipynb      (exit code 1 if any check fails)
"""
import ast
import re
import sys
from pathlib import Path

import nbformat

MAX_CODE_CELLS_AFTER_MARKDOWN = 2
EXPECTED_SECTIONS = 14
IDENTITY_LITERALS = {0, 1, 2, 100}
DATA_MODIFYING = re.compile(r"\.(merge|merge_asof|drop|rename|assign|query|dropna|fillna|pivot|melt|concat|read_csv|sort_values|reset_index)\(|"
                            r"^\s*[A-Za-z_][A-Za-z0-9_]*\[[^\]]+\]\s*=", re.MULTILINE)
PROSE_NUMBER = re.compile(r"(?<![\w.])[-−+]?\d+(?:[.,]\d+)?\s?(?:%|pts?\b|points?\b|\$)|(?<![\w./-])\d+[.,]\d+(?![\w.])")
INLINE_CODE_OR_URL = re.compile(r"`[^`]*`|https?://\S+")
HAND_BUILT_DATE = re.compile(r"Timestamp\(\s*[\w.]+\s*[-+]\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)|Timestamp\(\s*[\w.]+\s*,\s*\d+\s*,\s*\d+\s*\)")
THROWAWAY_NAMES = re.compile(r"^(df\d*|tmp\d*|temp\d*|foo|bar|var\d+|x\d+|data\d+|a\d+|b\d+)$")
CAMEL_CASE = re.compile(r"^[a-z]+[A-Z]")
FORBIDDEN_OUTPUT_COLUMNS = {"unit_code", "unit_id", "lease_id", "sUnitCode", "hUnit"}
FORBIDDEN_FILE_COLUMNS = FORBIDDEN_OUTPUT_COLUMNS | {"lease_start", "sign_date", "rent_contract", "rent_effective"}


class Finding:
    def __init__(self, check, cell_index, message):
        self.check, self.cell_index, self.message = check, cell_index, message

    def __str__(self):
        return f"[{self.check}] cell {self.cell_index}: {self.message}"


def numeric_literals(tree):
    """Every int/float literal of a code cell (booleans excluded)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            yield node.value, node.lineno


def space_padded_labels(tree):
    """String labels (dict keys, subscripts) with leading or trailing spaces."""
    for node in ast.walk(tree):
        keys = node.keys if isinstance(node, ast.Dict) else [node.slice] if isinstance(node, ast.Subscript) else []
        for key in keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str) and key.value != key.value.strip():
                yield key.value


def dead_names(trees):
    """Top-level names assigned somewhere in the notebook and never read anywhere."""
    assigned, read = {}, set()
    for index, tree in trees:
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        assigned.setdefault(name.id, index)
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                assigned.setdefault(node.name, index)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                read.add(node.id)
            elif isinstance(node, ast.Attribute):
                read.add(node.attr)
    return {name: index for name, index in assigned.items() if name not in read and not name.startswith("_")}


def lint(path):
    notebook = nbformat.read(path, as_version=4)
    findings, trees = [], []
    cells = list(notebook.cells)
    code_after_markdown, seen_markdown = 0, False
    for index, cell in enumerate(cells):
        if cell.cell_type == "markdown":
            seen_markdown, code_after_markdown = True, 0
            prose = INLINE_CODE_OR_URL.sub("", cell.source)
            for match in PROSE_NUMBER.finditer(prose):
                findings.append(Finding("prose-number", index, f"nombre tapé dans le texte : {match.group(0).strip()!r} (…{prose[max(0, match.start() - 40):match.end() + 20]}…)"))
            continue
        source = cell.source
        code_after_markdown += 1
        if DATA_MODIFYING.search(source) and (not seen_markdown or code_after_markdown > MAX_CODE_CELLS_AFTER_MARKDOWN):
            findings.append(Finding("markdown", index, f"étape qui modifie les données sans Markdown dans les {MAX_CODE_CELLS_AFTER_MARKDOWN} cellules précédentes"))
        try:
            tree = ast.parse(source)
        except SyntaxError as error:
            findings.append(Finding("syntax", index, str(error)))
            continue
        trees.append((index, tree))
        if not source.strip().startswith("# CONFIG"):
            for value, line in numeric_literals(tree):
                if value not in IDENTITY_LITERALS:
                    findings.append(Finding("magic-number", index, f"littéral {value!r} ligne {line} : {source.splitlines()[line - 1].strip()[:90]}"))
        for match in HAND_BUILT_DATE.finditer(source):
            findings.append(Finding("repeated-date", index, f"date construite à la main : {match.group(0)}"))
        for label in space_padded_labels(tree):
            findings.append(Finding("label-spaces", index, f"étiquette distinguée par des espaces : {label!r}"))
        if re.search(r"""["']site["']|sSite""", source) and re.search(r"unit_code|sUnitCode", source) and "wrong_" not in source and "key_collisions" not in source and "_RENAME" not in source:
            findings.append(Finding("wrong-key", index, "site + unit_code utilisé hors de la démonstration de la mauvaise clé"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                names.append(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.arg)):
                names.append(node.name if isinstance(node, ast.FunctionDef) else node.arg)
            for name in names:
                if THROWAWAY_NAMES.match(name) or (CAMEL_CASE.match(name) and not name.isupper()):
                    findings.append(Finding("naming", index, f"nom peu parlant ou non snake_case : {name}"))
        if cell.get("execution_count") is None and source.strip():
            findings.append(Finding("executed", index, "cellule non exécutée"))
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                findings.append(Finding("executed", index, f"erreur d'exécution : {output.get('ename')}"))
            text = output.get("text") or output.get("data", {}).get("text/plain") or ""
            header = "\n".join(text.splitlines()[:3])
            if any(re.search(rf"\b{re.escape(column)}\b", header) for column in FORBIDDEN_OUTPUT_COLUMNS) and "colonne source" not in text:
                findings.append(Finding("crm-output", index, "colonne d'identifiant d'unité dans une sortie"))
    for name, index in dead_names(trees).items():
        findings.append(Finding("dead-name", index, f"nom assigné mais jamais lu : {name}"))
    headings = [line for cell in cells if cell.cell_type == "markdown" for line in cell.source.splitlines() if line.startswith("## ")]
    if len(headings) < EXPECTED_SECTIONS:
        findings.append(Finding("structure", -1, f"{len(headings)} sections « ## » trouvées, {EXPECTED_SECTIONS} attendues"))
    for csv_file in (Path(path).parent / "outputs").glob("*.csv"):
        header = csv_file.read_text(encoding="utf-8").splitlines()[0]
        if any(column in header.split(",") for column in FORBIDDEN_FILE_COLUMNS):
            findings.append(Finding("crm-output", -1, f"{csv_file.name} contient des colonnes de niveau bail"))
    return findings, len(cells)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "equinoxe_hausse_2026.ipynb"
    results, cell_count = lint(target)
    for result in results:
        print(result)
    by_check = {}
    for result in results:
        by_check[result.check] = by_check.get(result.check, 0) + 1
    print(f"\n{cell_count} cellules, {len(results)} problème(s) : {by_check}")
    sys.exit(1 if results else 0)
