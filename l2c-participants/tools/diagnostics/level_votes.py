"""Counts only: storey vote table (shop storey position x plan storey position), positions as indices, no names."""
import sys
from collections import Counter, defaultdict
from pathlib import Path
from l2c_rebar.pipeline import run_project
from l2c_rebar.config import Config
import l2c_rebar.compare as cmp
if __name__ == "__main__":
    proj = Path(sys.argv[1])
    orig = cmp.Comparator._align_levels
    def spy(self, pairs):
        for t in {s.type_element for s, _, _ in pairs}:
            if t != "colonne": continue
            group = [(s, c) for s, c, _ in pairs if s.type_element == t and s.levels]
            levels = {l for s, _ in group for l in s.levels} | {l for _, cs in group for c in cs for l in c.levels}
            order = sorted(levels, key=cmp._level_key); rank = {l: i for i, l in enumerate(order)}
            plan_levels = sorted({rank[l] for _, cs in group for c in cs for l in c.levels})
            votes = defaultdict(Counter); n = Counter()
            for s, cs in group:
                n[s.levels] += 1
                for c in cs:
                    for l in c.levels:
                        if self._compare(c, s).conform: votes[s.levels][rank[l]] += 1
            print("order size", len(order), "| plan storey positions:", plan_levels)
            for key in sorted(votes, key=lambda k: rank[k[0]]):
                print("  shop storey positions", [rank[k] for k in key], "elements", n[key], "agree with plan position ->", dict(sorted(votes[key].items())))
        return orig(self, pairs)
    cmp.Comparator._align_levels = spy
    run_project(proj, Path("outputs") / proj.name, Config())
