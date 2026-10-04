"""Match plan elements to shop-drawing elements and classify each one.

Two matching tiers:

1. **By label** - an element named on both sides (C-12, S3, MR-2...) is paired
   through its normalised label.  When the plan gives several versions of the
   same label (one per storey in a column schedule), the storey read from the
   shop drawing's file name selects the version.
2. **By content** - callouts without a label (slab mats, typical details) are
   paired inside the same element type and storey by what they say, using their
   position on the sheet to break ties.

Only attributes present on both sides are compared, as the rules ask.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace

from .config import ELEMENT_TYPES, Config
from .models import AJOUTE, CONFORME, MANQUANT, NON_CONFORME, Bar, Ecart, Element, Result
from .extract.grid import bay_key

_SIZE_MM = {"10M": 11.3, "15M": 16.0, "20M": 19.5, "25M": 25.2, "30M": 29.9, "35M": 35.7, "45M": 43.7, "55M": 56.4}
_ATTR_LABEL = {"quantite": "Quantité", "espacement_mm": "Espacement", "longueur_mm": "Longueur", "diametre": "Diamètre"}


@dataclass
class BarComparison:
    ecarts: list[Ecart] = field(default_factory=list)
    checked: int = 0  # plan bars confirmed by the shop drawing
    unknown: int = 0  # plan bars found by diameter only (nothing else comparable)
    multiples: int = 0  # plan bars found in an exact multiple of their quantity (detail shared by several elements)

    @property
    def conform(self) -> bool:
        return not self.ecarts


def _rel(a, b, tol: int = 0) -> str:
    if a is None or b is None:
        return "?"
    return "=" if abs(a - b) <= tol else "!"


def _relations(p: Bar, s: Bar, cfg: Config) -> dict[str, str]:
    return {
        "quantite": _rel(p.quantite, s.quantite),
        # A plan in millimetres and a shop drawing in inches round differently (300 mm vs 12 in = 305 mm).
        "espacement_mm": _rel(p.espacement_mm, s.espacement_mm,
                              max(cfg.spacing_tol_mm, round(cfg.spacing_tol_ratio * (p.espacement_mm or 0)))),
        "longueur_mm": _rel(p.longueur_mm, s.longueur_mm, cfg.length_tol_mm),
    }


def _role_ok(p: Bar, s: Bar) -> bool:
    return p.role is None or s.role is None or p.role == s.role


def _ecart(attr: str, p: Bar, s: Bar | None, shop_value=None) -> Ecart:
    pv = getattr(p, attr)
    sv = shop_value if shop_value is not None else (getattr(s, attr) if s else None)
    label = _ATTR_LABEL[attr]
    if attr == "quantite":
        gravite = "critique" if sv < pv else "mineur"
        msg = f"{label} : plan {pv} barres {p.diametre or ''}, atelier {sv} ({'manque' if sv < pv else 'surplus'} {abs(pv - sv)})".replace("  ", " ").replace(" ,", ",")
    elif attr == "espacement_mm":
        gravite = "critique" if sv > pv else "mineur"
        msg = f"{label} : plan {p.diametre} @ {pv} mm, atelier @ {sv} mm ({'plus espacé' if sv > pv else 'plus serré'})"
    elif attr == "longueur_mm":
        gravite = "majeur" if sv < pv else "mineur"
        msg = f"{label} : plan {pv} mm, atelier {sv} mm ({sv - pv:+d} mm)"
    else:
        smaller = _SIZE_MM.get(str(sv), 0) < _SIZE_MM.get(str(pv), 0)
        gravite = "critique" if smaller else "majeur"
        msg = f"{label} : plan {pv}, atelier {sv} ({'plus petit' if smaller else 'plus gros'})"
    return Ecart(attribut=attr, plan=pv, atelier=sv, gravite=gravite, message=msg, plan_bbox=p.bbox,
                 atelier_bbox=s.bbox if s else None, plan_bar=p)


def _subset_sum(values: list[int], target: int) -> list[int] | None:
    """Indices of a smallest subset of `values` adding up to `target`, if there is one."""
    best: dict[int, list[int]] = {0: []}
    for k, v in enumerate(values):
        for total, idx in list(best.items()):
            t = total + v
            if t <= target and (t not in best or len(idx) + 1 < len(best[t])):
                best[t] = idx + [k]
    return best.get(target) or None


def compare_bars(plan_bars: list[Bar], shop_bars: list[Bar], cfg: Config) -> BarComparison:
    """Reconcile the bars of one plan element with those of one shop element."""
    out = BarComparison()
    used: set[int] = set()

    # 1. A shop bar agreeing on everything that can be compared.  Counted bars
    #    pair one-to-one; a spacing-only plan bar may be honoured by a bar already paired.
    pending: list[Bar] = []
    for p in plan_bars:
        exact = []
        for i, s in enumerate(shop_bars):
            if (p.diametre is not None and s.diametre != p.diametre) or (p.quantite is not None and i in used):
                continue
            r = _relations(p, s, cfg)
            if "!" not in r.values() and "=" in r.values():
                exact.append((i in used, not _role_ok(p, s), list(r.values()).count("?"), i))
        if exact:
            used.add(min(exact)[3])
            out.checked += 1
        else:
            pending.append(p)

    for p in pending:
        same = [(i, s) for i, s in enumerate(shop_bars) if p.diametre is None or s.diametre == p.diametre]
        rels = {i: _relations(p, s, cfg) for i, s in same}
        free = [(i, s) for i, s in same if i not in used or p.quantite is None]

        # 2. The shop drawing splits the plan quantity over several marks.
        pool = [(i, s) for i, s in same if s.quantite is not None and i not in used and _role_ok(p, s)]
        total = sum(s.quantite for _, s in pool)
        if p.quantite is not None and len(pool) > 1:
            subset = _subset_sum([s.quantite for _, s in pool], p.quantite)
            if subset:
                used.update(pool[k][0] for k in subset)
                out.checked += 1
                continue

        differing = [(i, s) for i, s in free if "!" in rels[i].values()]
        if differing:
            # 3. Same diameter, different numbers: report against the closest shop bar.
            def distance(it):
                i, s = it
                r = rels[i]
                d = 0.0
                if r["quantite"] == "!":
                    d += abs(s.quantite - p.quantite) / max(p.quantite, 1)
                if r["espacement_mm"] == "!":
                    d += abs(s.espacement_mm - p.espacement_mm) / max(p.espacement_mm, 1)
                if r["longueur_mm"] == "!":
                    d += abs(s.longueur_mm - p.longueur_mm) / max(p.longueur_mm, 1)
                return (i in used, not _role_ok(p, s), list(r.values()).count("!"), d)

            i, s = min(differing, key=distance)
            r = rels[i]
            used.add(i)
            only_quantity = r["quantite"] == "!" and r["espacement_mm"] != "!" and r["longueur_mm"] != "!"
            if only_quantity and p.quantite and s.quantite % p.quantite == 0 and 2 <= s.quantite // p.quantite <= 12:
                # Twice, three times... the plan quantity: one detail drawn for that many
                # identical elements, with the total written on it.  Not a discrepancy.
                out.multiples += 1
                continue
            if r["quantite"] == "!" and len(pool) > 1 and abs(total - p.quantite) < abs(s.quantite - p.quantite):
                out.ecarts.append(_ecart("quantite", p, s, shop_value=total))
                used.update(j for j, _ in pool)
            elif r["quantite"] == "!":
                out.ecarts.append(_ecart("quantite", p, s))
            for attr in ("espacement_mm", "longueur_mm"):
                if r[attr] == "!":
                    out.ecarts.append(_ecart(attr, p, s))
            continue

        if free:
            # 4. Diameter present but nothing else to compare (plan gives a
            #    spacing, shop a quantity, or the reverse).
            out.unknown += 1
            continue

        # 5. No bar of that diameter: same numbers under another diameter?
        swapped = [
            (i, s) for i, s in enumerate(shop_bars)
            if i not in used and s.diametre and "!" not in (r := _relations(p, s, cfg)).values() and "=" in r.values()
        ]
        if swapped:
            i, s = swapped[0]
            used.add(i)
            out.ecarts.append(_ecart("diametre", p, s))
        elif p.quantite is None:
            # Ties and distributed bars given by spacing are often covered by a general
            # note on the shop drawing: nothing to compare, which is not a discrepancy.
            out.unknown += 1
        else:
            out.ecarts.append(
                Ecart(attribut="absence", plan=p.describe(), atelier=None, gravite="majeur",
                      message=f"Barres {p.describe()} du plan introuvables dans le dessin d'atelier", plan_bbox=p.bbox,
                      plan_bar=p)
            )
    return out


# ---------------------------------------------------------------- matching

def _level_key(level: str) -> float:
    """Height order of storey names: foundations, basements, ground floor, storeys, roof."""
    if level == "FDN":
        return -1000.0
    if level == "TREFOND":
        return -500.0
    if level.startswith("SS"):
        return -10.0 * int(level[2:] or 1)
    if level == "RDC":
        return 10.0
    if level == "MEZZ":
        return 15.0
    if level == "TOIT":
        return 10000.0
    try:
        return 10.0 * float(level)
    except ValueError:
        return 5000.0


def _levels_compatible(a: Element, b: Element) -> bool:
    return not a.levels or not b.levels or bool(set(a.levels) & set(b.levels))


def _result(statut: str, plan: Element | None, shop: Element | None, feuillet: str, comp: BarComparison | None,
            methode: str, factor: float, note: str = "", ocr_factor: float = 0.8) -> Result:
    ref = plan or shop
    assert ref is not None
    # Element confidences already carry the OCR score of their lines; `ocr_factor` adds the
    # risk of a misread the score does not see (a 2 read as a 5 with full confidence).
    conf = min(e.conf for e in (plan, shop) if e is not None) * factor
    if any(e is not None and e.ocr for e in (plan, shop)):
        conf *= ocr_factor
        note = (note + " " if note else "") + "Texte lu par OCR."
    if comp is not None and comp.multiples:
        conf *= 0.85
        note = (note + " " if note else "") + "Quantité d'atelier multiple de celle du plan : détail commun à plusieurs éléments."
    if comp is not None and comp.conform and comp.checked == 0:
        conf *= 0.6  # nothing but the diameter could be verified
        note = (note + " " if note else "") + "Seul le diamètre a pu être vérifié."
    return Result(statut=statut, type_element=ref.type_element, element=ref.element, feuillet=feuillet, plan=plan,
                  atelier=shop, ecarts=list(comp.ecarts) if comp else [], confiance=round(max(0.05, min(conf, 0.99)), 2),
                  methode=methode, note=note.strip(), verifiees=comp.checked if comp else 0)


@dataclass
class Coverage:
    """What could not be compared at all, reported instead of flooding the lists."""

    types_sans_atelier: list[str] = field(default_factory=list)
    feuillets_sans_atelier: list[str] = field(default_factory=list)
    totaux: dict[str, dict] = field(default_factory=dict)  # shared details: per element or totals
    level_shift: dict[str, dict] = field(default_factory=dict)
    lecture: dict[str, str] = field(default_factory=dict)  # per type: sheets read as a schedule or as details
    atelier_non_apparies: int = 0
    non_verifies: int = 0  # plan callouts that could be neither confirmed nor contradicted
    feuillets_lecture_partielle: dict[str, str] = field(default_factory=dict)  # sheet -> found/total
    reperes_non_fiables: list[str] = field(default_factory=list)  # types where pairing by name was abandoned
    ocr_lignes_recollees: list[str] = field(default_factory=list)  # types whose OCR rows were joined back
    accord_par_type: dict[str, float] = field(default_factory=dict)  # share of bars confirmed in pairs made by name


class Comparator:
    def __init__(self, plan: list[Element], shop: list[Element], cfg: Config):
        self.cfg = cfg
        keep = (lambda e: not e.bordereau and (cfg.compare_other_types or e.type_element in ELEMENT_TYPES))
        self.plan = [e for e in plan if keep(e)]
        self.shop = [e for e in shop if keep(e)]
        self.coverage = Coverage()
        self.results: list[Result] = []
        self._typical: set[int] = set()  # shop elements that satisfied a spacing-only plan callout
        self._cover: dict[tuple[str, bool], set[str]] = {}
        self.unpaired_shop = 0  # unnamed shop callouts left without a plan twin
        self._level_map: dict[tuple[str, tuple[str, ...]], str] = {}  # shop storey name -> plan storey
        self._totals: dict[str, bool] = {}  # element types whose shared details list totals
        self._level_order: dict[str, list[str]] = {}
        self._by_page: dict[tuple[str, int], list[Element]] = defaultdict(list)
        for s in self.shop:
            self._by_page[(s.path or s.fichier, s.page)].append(s)
        self._norm = self._normalised_positions(self.plan + self.shop)
        self._sheets = self._sheet_index()

    # -- helpers ---------------------------------------------------------
    def _result(self, *args, **kwargs) -> Result:
        return _result(*args, ocr_factor=self.cfg.ocr_confidence_factor, **kwargs)

    @staticmethod
    def _normalised_positions(elements: list[Element]) -> dict[int, tuple[float, float]]:
        """Position of each element inside the drawn area of its page, in [0, 1]."""
        boxes: dict[tuple[str, int], list[float]] = {}
        for e in elements:
            b = boxes.setdefault((e.path or e.fichier, e.page), [e.x, e.y, e.x, e.y])
            b[0], b[1], b[2], b[3] = min(b[0], e.x), min(b[1], e.y), max(b[2], e.x), max(b[3], e.y)
        out = {}
        for e in elements:
            x0, y0, x1, y1 = boxes[(e.path or e.fichier, e.page)]
            out[id(e)] = ((e.x - x0) / max(x1 - x0, 1.0), (e.y - y0) / max(y1 - y0, 1.0))
        return out

    def _dist(self, a: Element, b: Element) -> float:
        (ax, ay), (bx, by) = self._norm[id(a)], self._norm[id(b)]
        return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5

    def _sheet_index(self) -> dict[str, list[tuple[str, tuple[str, ...]]]]:
        index: dict[str, list[tuple[str, tuple[str, ...]]]] = defaultdict(list)
        seen = set()
        for e in self.plan:
            if e.feuillet not in seen:
                seen.add(e.feuillet)
                index[e.type_element].append((e.feuillet, e.levels))
        return index

    def _plan_sheet_for(self, shop: Element) -> str:
        """Plan sheet under which a shop-only element is reported."""
        sheets = self._sheets.get(shop.type_element, [])
        for feuillet, levels in sheets:
            if shop.levels and set(levels) & set(shop.levels):
                return feuillet
        return sheets[0][0] if sheets else "HORS-PLAN"

    # -- tier 1: labels --------------------------------------------------
    def _align_levels(self, pairs: list[tuple[Element, list[Element], float]]) -> None:
        """Which plan storey does each shop storey name stand for?

        Detailers name lifts their own way: "NIV 2@3", the slab below or the slab
        above, a 13th floor skipped.  Most of a shop drawing agrees with its plan,
        so shop storey names are mapped to plan storeys, in order, so as to agree as
        often as possible, looking no further than two storeys from each name.
        """
        for type_element in {s.type_element for s, _, _ in pairs}:
            group = [(s, cands) for s, cands, _ in pairs if s.type_element == type_element and s.levels]
            levels = {l for s, _ in group for l in s.levels} | {l for _, cands in group for c in cands for l in c.levels}
            order = sorted(levels, key=_level_key)
            self._level_order[type_element] = order
            rank = {level: i for i, level in enumerate(order)}
            votes: dict[tuple[str, ...], Counter] = defaultdict(Counter)
            for s, cands in group:
                for c in cands:
                    for level in c.levels:
                        near = min(abs(rank[level] - rank[k]) for k in s.levels) <= 2
                        if near and self._compare(c, s).conform:
                            votes[s.levels][level] += 1
            if not votes:
                continue
            sizes = Counter(s.levels for s, _ in group)
            plan_levels = sorted({l for _, cands in group for c in cands for l in c.levels}, key=rank.__getitem__)
            keys = sorted(votes, key=lambda k: (rank[k[0]], rank[k[-1]]))
            # Shop storeys keep their order: give each a plan storey, never stepping back
            # down, so that the drawings agree as often as possible.  Two shop storeys may
            # share a plan storey (FDN and FDN@RDC both detail the lowest lift), but at a
            # cost, so that it takes evidence to do so.
            table: list[dict[str, tuple[float, str | None]]] = []
            for i, key in enumerate(keys):
                row: dict[str, tuple[float, str | None]] = {}
                for target in plan_levels:
                    own = votes[key][target] - 0.01 * min(abs(rank[target] - rank[k]) for k in key)
                    if i == 0:
                        row[target] = (own, None)
                        continue
                    options = [(score + own - (0.25 * sizes[key] if prev == target else 0.0), prev)
                               for prev, (score, _) in table[-1].items() if rank[prev] <= rank[target]]
                    if options:
                        row[target] = max(options)
                table.append(row)
            target: str | None = max(table[-1], key=lambda t: table[-1][t][0])
            renamed = 0
            for i in range(len(keys) - 1, -1, -1):
                assert target is not None
                self._level_map[(type_element, keys[i])] = target
                renamed += target not in keys[i]
                target = table[i][target][1]
            self.coverage.level_shift[type_element] = {"niveaux_atelier": len(votes), "renommes": renamed}

    def _plan_levels(self, shop: Element) -> tuple[str, ...]:
        """Storeys of the plan that a shop element stands for (see `_align_levels`)."""
        mapped = self._level_map.get((shop.type_element, shop.levels))
        return (mapped,) if mapped else shop.levels

    def _shop_bars(self, shop: Element) -> list[Bar]:
        """Bars of a shop element, per element: a detail drawn for several elements
        lists totals when the project follows that habit (`self._totals`)."""
        n = shop.multiplicity
        if n <= 1 or not self._totals.get(shop.type_element):
            return shop.bars
        return [replace(b, quantite=b.quantite // n) if b.quantite and b.quantite % n == 0 else b for b in shop.bars]

    def _compare(self, plan: Element, shop: Element) -> BarComparison:
        return compare_bars(plan.bars, self._shop_bars(shop), self.cfg)

    def _pick(self, shop: Element, cands: list[Element]) -> tuple[Element, float, str]:
        shop_levels = self._plan_levels(shop)
        if len(cands) == 1:
            ok = not cands[0].levels or not shop_levels or bool(set(cands[0].levels) & set(shop_levels))
            return cands[0], (1.0 if ok else 0.6), ("" if ok else "Niveau du plan différent de celui du dessin d'atelier.")
        level_hits = [c for c in cands if shop_levels and set(c.levels) & set(shop_levels)]
        if len(level_hits) == 1:
            return level_hits[0], 1.0, ""
        if len(level_hits) > 1:
            return min(level_hits, key=lambda c: len(self._compare(c, shop).ecarts)), 0.8, ""
        best = min(cands, key=lambda c: len(self._compare(c, shop).ecarts))
        return best, 0.7, "Niveau non identifié : comparé à la version du plan la plus proche."

    def _match_labels(self) -> tuple[list[Element], list[Element]]:
        index: dict[tuple[str, str], list[Element]] = defaultdict(list)
        by_label: dict[str, list[Element]] = defaultdict(list)
        between: dict[tuple[str, tuple[str, str]], list[Element]] = defaultdict(list)
        for p in self.plan:
            if p.label:
                index[(p.type_element, p.label)].append(p)
                by_label[p.label].append(p)
                if "/" in p.label and (key := bay_key(p.label)):
                    between[(p.type_element, key)].append(p)  # a column between two labelled lines

        pairs: list[tuple[Element, list[Element], float]] = []
        loose_shop: list[Element] = []
        for s in self.shop:
            if not s.label:
                loose_shop.append(s)
                continue
            cands, factor = index.get((s.type_element, s.label)), 1.0
            if not cands:
                other = by_label.get(s.label, [])
                if other and len({p.type_element for p in other}) == 1:
                    cands, factor = other, 0.8  # same label filed under another element type
            if not cands and "." in s.label and (key := bay_key(s.label)):
                # The shop drawing names an intermediate line ("B.1-12") that the plan leaves
                # unlabelled: the plan element standing between B and the next line, on 12.
                cands, factor = between.get((s.type_element, key)), 0.9
            if cands:
                pairs.append((s, cands, factor))
            else:
                loose_shop.append(s)

        # A detail drawn once for several elements ("B-12, B-13") may list the bars of one
        # element or the total for all of them; keep the reading that agrees most often.
        for type_element in {s.type_element for s, _, _ in pairs}:
            multi = [(s, cands) for s, cands, _ in pairs if s.type_element == type_element and s.multiplicity > 1]
            tally = {}
            for totals in (False, True):
                self._totals[type_element] = totals
                tally[totals] = sum(any(self._compare(c, s).conform for c in cands) for s, cands in multi)
            self._totals[type_element] = tally[True] > tally[False]
            if multi:
                self.coverage.totaux[type_element] = {"details_multiples": len(multi), "accords_par_element": tally[False],
                                                      "accords_totaux": tally[True]}

        self._align_levels(pairs)

        shop_by_name: dict[tuple[str, str], list[Element]] = defaultdict(list)
        for s in self.shop:
            if s.label:
                shop_by_name[(s.type_element, s.label)].append(s)

        used: set[int] = set()
        for s, cands, factor in pairs:
            p, f2, note = self._pick(s, cands)
            used.add(id(p))
            comp = self._compare(p, s)
            if s.multiplicity > 1 and self._totals.get(s.type_element):
                note = (note + " " if note else "") + f"Détail d'atelier commun à {s.multiplicity} éléments : quantités divisées."
            if self._recover_nearby(s, comp):
                f2 *= 0.8
                note = (note + " " if note else "") + "Barres retrouvées près du repère, hors du groupe apparié."
            if self._recover_adjacent_storey(s, comp, shop_by_name):
                f2 *= 0.85
                note = (note + " " if note else "") + "Barres détaillées au niveau voisin (barres continues sur deux niveaux)."
            if comp.ecarts and all(e.attribut == "absence" for e in comp.ecarts):
                f2 *= 0.75  # nothing contradicts the plan; bars may simply be drawn elsewhere
            statut = CONFORME if comp.conform else NON_CONFORME
            self.results.append(self._result(statut, p, s, p.feuillet, comp, "repere", factor * f2, note))

        shop_labels = {(s.type_element, s.label) for s in self.shop if s.label}
        # Share of the named plan elements of each sheet that the shop drawings account for.
        seen: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for p in self.plan:
            if p.label:
                seen[p.feuillet][0] += id(p) in used
                seen[p.feuillet][1] += 1
        loose_plan = []
        for p in self.plan:
            if id(p) in used:
                continue
            if p.label and (p.type_element, p.label) in shop_labels:
                # The name is detailed, but not this version (another storey).
                found, total = seen[p.feuillet]
                if not self._level_covered(p, True):
                    continue
                if found < self.cfg.min_sheet_coverage * total:
                    # Most of the sheet is unaccounted for: the shop drawings of that storey were
                    # not provided or could not be read.  Said once, not element by element.
                    self.coverage.feuillets_lecture_partielle[p.feuillet] = f"{found}/{total}"
                    continue
                self.results.append(self._result(MANQUANT, p, None, p.feuillet, None, "repere", 0.7,
                                            "Repère présent à l'atelier, mais pas pour ce niveau."))
                continue
            loose_plan.append(p)
        return loose_plan, loose_shop

    def _recover_adjacent_storey(self, s: Element, comp: BarComparison,
                                 shop_by_name: dict[tuple[str, str], list[Element]]) -> bool:
        """Drop "bars not found" when the same element carries those bars one storey
        up or down: vertical bars are often detailed once for two storeys."""
        missing = [e for e in comp.ecarts if e.attribut == "absence" and e.plan_bar is not None]
        if not missing or len(s.levels) != 1 or not s.label:
            return False
        order = self._level_order.get(s.type_element, [])
        if s.levels[0] not in order:
            return False
        here = order.index(s.levels[0])
        neighbours = [b for other in shop_by_name.get((s.type_element, s.label), ()) if other is not s
                      and len(other.levels) == 1 and other.levels[0] in order
                      and abs(order.index(other.levels[0]) - here) == 1 for b in other.bars]
        recovered = False
        for e in missing:
            again = compare_bars([e.plan_bar], neighbours, self.cfg) if neighbours else None
            if again is not None and again.conform and again.checked:
                comp.ecarts.remove(e)
                comp.unknown += 1
                recovered = True
        return recovered

    def _recover_nearby(self, s: Element, comp: BarComparison) -> bool:
        """Drop "bars not found" discrepancies when the plan bars are drawn close to
        the shop element as loose callouts that the layout step left unnamed.
        Callouts owned by a neighbouring element never excuse a discrepancy, and a
        diameter swap (same count, other size) is never excused: it is too specific
        to be a layout accident.  True if any was dropped."""
        suspects = [e for e in comp.ecarts if e.attribut == "absence"]
        if not suspects:
            return False
        reach = 1.5 * self.cfg.label_radius
        nearby = [b for other in self._by_page.get((s.path or s.fichier, s.page), ())
                  if other is not s and (not other.label or self.cfg.recover_from_neighbours)
                  for b in other.bars
                  if abs((b.bbox[0] + b.bbox[2]) / 2 - s.x) <= 1.6 * reach and abs((b.bbox[1] + b.bbox[3]) / 2 - s.y) <= reach]
        recovered = False
        for e in suspects:
            if e.plan_bar is None or not nearby:
                continue
            again = compare_bars([e.plan_bar], nearby, self.cfg)
            if again.conform and (again.checked or e.attribut == "absence"):
                comp.ecarts.remove(e)
                comp.unknown += 1
                recovered = True
        return recovered

    def _level_covered(self, p: Element, aligned: bool = False) -> bool:
        """True when some shop drawing of that type covers the element's storey.

        With `aligned`, shop storey names are first translated to plan storeys.
        """
        if not p.levels:
            return True
        key = (p.type_element, aligned)
        if key not in self._cover:
            levels: set[str] = set()
            for s in self.shop:
                if s.type_element != p.type_element:
                    continue
                if not s.levels:
                    levels.add("*")
                else:
                    levels.update(self._plan_levels(s) if aligned else s.levels)
            self._cover[key] = levels
        covered = self._cover[key]
        return "*" in covered or bool(covered & set(p.levels))

    # -- tier 2: content -------------------------------------------------
    def _match_content(self, loose_plan: list[Element], loose_shop: list[Element]) -> None:
        cfg = self.cfg
        shop_by_type: dict[str, list[Element]] = defaultdict(list)
        for s in loose_shop:
            shop_by_type[s.type_element].append(s)
        by_dia: dict[tuple[str, str], list[Element]] = defaultdict(list)
        for s in loose_shop:
            for d in {b.diametre for b in s.bars if b.diametre}:
                by_dia[(s.type_element, d)].append(s)

        used: set[int] = set()
        for p in sorted(loose_plan, key=lambda e: -e.conf):
            if not shop_by_type.get(p.type_element):
                continue  # no shop drawing of that type: reported once, in the coverage notes
            if not self._level_covered(p):
                continue
            seen: set[int] = set()
            cands: list[Element] = []
            for d in {b.diametre for b in p.bars if b.diametre}:
                for s in by_dia.get((p.type_element, d), ()):
                    if id(s) not in seen and _levels_compatible(p, s):
                        seen.add(id(s))
                        cands.append(s)

            if self._match_position(p, shop_by_type[p.type_element]):
                continue
            if all(b.diametre is None for b in p.bars):
                # A bare count ("12(6)") with no twin at its place: without a bar size there is
                # nothing to search the rest of the storey for.  Counted, not reported.
                self.coverage.non_verifies += 1
                continue

            # A counted callout ("10-20M") pairs with one shop callout; a spacing-only
            # one ("15M @ 300 typ.") is honoured by every shop callout that agrees.
            consumes = any(b.quantite is not None for b in p.bars)
            conform: tuple[tuple, Element, BarComparison] | None = None
            mismatch: tuple[tuple, Element, BarComparison] | None = None
            for s in cands:
                comp = compare_bars(p.bars, s.bars, cfg)
                taken = consumes and id(s) in used
                d = self._dist(p, s)
                if comp.conform and comp.checked:
                    if not consumes:
                        self._typical.add(id(s))
                    key = (taken, d)
                    if conform is None or key < conform[0]:
                        conform = (key, s, comp)
                elif comp.ecarts and all(e.attribut != "absence" for e in comp.ecarts) and not taken:
                    key = (len(comp.ecarts), d)
                    if d <= cfg.position_tol and (mismatch is None or key < mismatch[0]):
                        mismatch = (key, s, comp)

            label_factor = 0.9 if p.label else 0.8
            if conform is not None:
                _, s, comp = conform
                if consumes:
                    used.add(id(s))
                self.results.append(self._result(CONFORME, p, s, p.feuillet, comp, "signature", label_factor))
            elif mismatch is not None:
                _, s, comp = mismatch
                used.add(id(s))
                self.results.append(self._result(NON_CONFORME, p, s, p.feuillet, comp, "signature", 0.7 * label_factor,
                                            "Apparié par contenu et position : à confirmer sur le feuillet."))
            else:
                self.results.append(self._result(MANQUANT, p, None, p.feuillet, None, "signature", 0.6 * label_factor))

        # "Added in the shop drawings" is said of named elements only.  A shop
        # drawing calls out every bar while a plan relies on typical notes, so an
        # unnamed shop callout without a plan twin is expected, not a finding.
        plan_types = {p.type_element for p in self.plan}
        for s in loose_shop:
            if id(s) in used or id(s) in self._typical or s.type_element not in plan_types:
                continue
            if s.label:
                self.results.append(self._result(AJOUTE, None, s, self._plan_sheet_for(s), None, "signature", 0.7))
            else:
                self.unpaired_shop += 1

    # -- tier 2a: place on the grid --------------------------------------
    def _bays(self, p: Element, s: Element) -> float:
        """Distance between a plan element and a shop element, in grid bays."""
        assert p.grid is not None and s.grid is not None
        return (((p.grid[0] - s.grid[0]) / p.grid[2]) ** 2 + ((p.grid[1] - s.grid[1]) / p.grid[3]) ** 2) ** 0.5

    def _match_position(self, p: Element, pool: list[Element]) -> bool:
        """Confirm a plan callout with the shop callouts drawn at the same place on the grid.

        Both drawings show the same floor at their own scale; the grid lines give
        every callout a place that is the same on both.  The shop callouts within
        `position_bays` of the plan callout are pooled: if they carry what the plan
        asks for, the callout is compliant.

        Nothing else is concluded from position.  On the development projects only
        about one slab callout in five has its twin within half a bay (a detailer
        writes a bar where it starts, an engineer where it is needed), so a
        different number nearby is nearly always another bar, not an error.
        """
        if p.grid is None:
            return False
        near = [s for s in pool if s.grid is not None and _levels_compatible(p, s)
                and self._bays(p, s) <= self.cfg.position_bays]
        if not near:
            return False
        comp = compare_bars(p.bars, [b for s in near for b in s.bars], self.cfg)
        if not (comp.conform and comp.checked):
            return False
        self._typical.update(id(s) for s in near)
        partner = min(near, key=lambda s: self._bays(p, s))
        self.results.append(self._result(CONFORME, p, partner, p.feuillet, comp, "position", 0.85))
        return True

    # -- driver ----------------------------------------------------------
    def run(self) -> list[Result]:
        shop_types = {s.type_element for s in self.shop}
        self.coverage.types_sans_atelier = sorted({p.type_element for p in self.plan} - shop_types)
        self.coverage.feuillets_sans_atelier = sorted(
            {p.feuillet for p in self.plan if p.type_element not in shop_types or not self._level_covered(p)}
        )
        loose_plan, loose_shop = self._match_labels()
        self._match_content(loose_plan, loose_shop)
        self.coverage.atelier_non_apparies = self.unpaired_shop

        order = {NON_CONFORME: 0, MANQUANT: 1, AJOUTE: 2, CONFORME: 3}
        self.results.sort(key=lambda r: (r.feuillet, order[r.statut], -r.confiance, r.element))
        counters: Counter = Counter()
        prefix = {NON_CONFORME: "NC", MANQUANT: "MQ", AJOUTE: "AJ", CONFORME: "OK"}
        for r in self.results:
            counters[r.statut] += 1
            r.id = f"{prefix[r.statut]}-{counters[r.statut]:04d}"
        return self.results


def compare_project(plan: list[Element], shop: list[Element], cfg: Config) -> tuple[list[Result], Coverage]:
    comparator = Comparator(plan, shop, cfg)
    return comparator.run(), comparator.coverage


def summarize(results: list[Result]) -> dict[str, dict[str, int]]:
    """Counts per plan sheet: the table at the heart of the report."""
    table: dict[str, dict[str, int]] = {}
    for r in results:
        row = table.setdefault(r.feuillet, {CONFORME: 0, NON_CONFORME: 0, MANQUANT: 0, AJOUTE: 0, "a_valider": 0})
        row[r.statut] += 1
        row["a_valider"] += r.a_valider
    return dict(sorted(table.items()))
