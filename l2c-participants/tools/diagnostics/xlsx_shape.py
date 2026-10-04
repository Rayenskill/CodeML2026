"""Structure of the spreadsheet only: every digit becomes 9, every word becomes A (a = single letter),
bar sizes become #M.  No value, name or sentence from the file is printed."""
import re, sys
from collections import Counter
import openpyxl
SIZE = re.compile(r"(?<![0-9A-Za-z.,/])(?:10|15|20|25|30|35|45|55)\s?M(?![A-Za-z])")
def mask(v):
    if v is None: return ""
    t = SIZE.sub("\x00", str(v))
    t = re.sub(r"[A-Za-zÀ-ÿ]+", lambda m: "A" if len(m.group()) > 1 else "a", t)
    t = re.sub(r"\d+", lambda m: "9" * min(len(m.group()), 3), t)
    t = re.sub(r"\s+", " ", t).strip().replace("\x00", "#M")
    return t if len(t) <= 40 else t[:40] + "~"
wb = openpyxl.load_workbook(sys.argv[1], data_only=True)
print("sheets:", len(wb.worksheets))
for ws in wb.worksheets:
    rows = [r for r in ws.iter_rows(values_only=True) if any(c is not None and str(c).strip() for c in r)]
    ncol = max((len(r) for r in rows), default=0)
    print(f"sheet: {ws.max_row} x {ws.max_column} cells, {len(rows)} non-empty rows, merged ranges: {len(ws.merged_cells.ranges)}")
    for i, r in enumerate(rows[:3]):
        print(f"  row {i + 1} masked:", [mask(c) for c in r[:ncol]])
    for c in range(ncol):
        vals = [r[c] for r in rows[1:] if c < len(r) and r[c] is not None and str(r[c]).strip()]
        shapes = Counter(mask(v) for v in vals)
        kinds = Counter(type(v).__name__ for v in vals)
        print(f"  column {c + 1}: filled={len(vals)} types={dict(kinds)} distinct values={len(set(map(str, vals)))} shapes={shapes.most_common(8)}")
