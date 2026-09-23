"""Auswertung der Dynasearch-Demo: Dynasearch mit Neustarts gegen sequentielles bestes-Verbesserung-2-opt mit
Neustarts, gleiches Bewertungsbudget, voller Rescan (wie Tabu Search/GRASP - Dynasearch ist nicht Kandidatenlisten-
nativ). Sweeps, Vergleichstabellen, Kettenstreuung.

Der Abstand zur Schranke ist der Abstand zu einer *unteren* Schranke der kürzesten Tour (1-Baum, Held-Karp). Ein
Vorschlag ist ein geprüftes (i,j)-Paar aus der vollen Delta-Matrix - dieselbe Einheit für beide Modi, da beide pro
Iteration dieselbe volle Matrix berechnen (nur die Zahl der daraus abgeleiteten, gleichzeitig angewandten Züge
unterscheidet sich)."""

import time
from dataclasses import dataclass, replace
from functools import lru_cache

import numpy as np

import dyna_algorithm as DA
import dyna_constants as C
import dyna_scenario as S
import dyna_tour as T


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    cluster_share: int = C.DEFAULT_BALLUNG
    seed: int = C.DEFAULT_SEED
    mode: str = C.DEFAULT_MODE
    budget: int = C.DEFAULT_BUDGET
    chain_seed: int = C.DEFAULT_CHAIN_SEED


@lru_cache(maxsize=256)
def instance(n, cluster_share, seed):
    inst = S.generate(n, cluster_share, seed)
    return inst, T.dist_matrix(inst.xy)


@lru_cache(maxsize=256)
def reference_bound(n, cluster_share, seed):
    inst, D = instance(n, cluster_share, seed)
    ref = T.descend(D, T.nearest_neighbor_tour(D), "2opt+oropt", "best", keep_steps=False)
    return T.held_karp_bound(D, ref.length, C.BOUND_ITERATIONS)


def _one_descend(D, start, mode, max_evaluations):
    if mode == "dynasearch":
        r = DA.dynasearch_descend(D, start, keep_steps=False, max_evaluations=max_evaluations)
        return r.tour, r.length, r.evaluations
    r = T.descend(D, start, "2opt", "best", keep_steps=False, max_evaluations=max_evaluations)
    return r.tour, r.length, r.evaluations


def restarts(D, mode, budget, seed):
    """Wiederholte Abstiege (gewählter Modus) aus zufälligen Startlösungen, bis das Budget erschöpft ist; der
    erste Abstieg läuft immer zu Ende. Gibt (beste Tour, Neustarts, verbrauchte Bewertungen) zurück."""
    rng = np.random.default_rng(seed)
    used, starts, best_tour, best_length = 0, 0, None, None
    while used < budget or best_tour is None:
        cap = None if best_tour is None else budget - used
        tour, length, spent = _one_descend(D, T.random_tour(len(D), rng), mode, cap)
        used += spent
        starts += 1
        if best_length is None or length < best_length:
            best_length, best_tour = length, tour
    return best_tour, starts, used


@dataclass
class Analysis:
    settings: Settings
    inst: object
    D: np.ndarray
    bound: float
    run_tour: np.ndarray
    run_starts: int
    seconds: float
    other_tour: np.ndarray             # der jeweils ANDERE Modus, gleiches Budget - die Vergleichsgröße
    other_starts: int
    other_seconds: float
    single_length: float               # ein einzelner Abstieg im gewählten Modus (kein Neustart), zum Vergleich
    single_evaluations: int
    single_iterations: int             # Rescans (Dynasearch) bzw. Einzelzüge (sequentiell) des einzelnen Abstiegs
    crossings_end: int

    def gap_of(self, length):
        return 100.0 * (length - self.bound) / self.bound

    @property
    def gap(self):
        return self.gap_of(T.tour_length(self.run_tour, self.D))

    @property
    def other_gap(self):
        return self.gap_of(T.tour_length(self.other_tour, self.D))

    @property
    def single_gap(self):
        return self.gap_of(self.single_length)


OTHER_MODE = {"dynasearch": "sequential", "sequential": "dynasearch"}


def analyse(settings, with_baselines=True):
    inst, D = instance(settings.n, settings.cluster_share, settings.seed)
    bound = reference_bound(settings.n, settings.cluster_share, settings.seed)
    t0 = time.perf_counter()
    run_tour, run_starts, _ = restarts(D, settings.mode, settings.budget, settings.chain_seed)
    seconds = time.perf_counter() - t0
    other_tour = None
    other_starts = 0
    other_seconds = 0.0
    single_length = single_evaluations = single_iterations = None
    if with_baselines:
        t0 = time.perf_counter()
        other_tour, other_starts, _ = restarts(D, OTHER_MODE[settings.mode], settings.budget, settings.chain_seed)
        other_seconds = time.perf_counter() - t0
        start = T.random_tour(len(D), np.random.default_rng(settings.chain_seed))
        if settings.mode == "dynasearch":
            r = DA.dynasearch_descend(D, start, keep_steps=False)
            single_length, single_evaluations, single_iterations = r.length, r.evaluations, r.n_iterations
        else:
            r = T.descend(D, start, "2opt", "best", keep_steps=False)
            single_length, single_evaluations, single_iterations = r.length, r.evaluations, r.n_moves
    return Analysis(settings, inst, D, bound, run_tour, run_starts, seconds, other_tour, other_starts, other_seconds,
                     single_length, single_evaluations, single_iterations, T.count_crossings(inst.xy, run_tour))


# --- Urteil ------------------------------------------------------------------------------------------------------------------------------------

WIN_MARGIN = 0.3
LOSE_MARGIN = 0.3


def verdict(a):
    """Code: beats_hc (Dynasearch klar besser als sequentiell, gleiches Budget), hc_wins (sequentiell besser),
    comparable. Immer relativ zum gewählten Modus formuliert: bei mode='sequential' ist 'beats_hc' ein Sieg des
    SEQUENTIELLEN Verfahrens gegen Dynasearch - siehe `a.settings.mode` für die Zuordnung."""
    if a.gap <= a.other_gap - WIN_MARGIN:
        return "beats_hc"
    if a.other_gap <= a.gap - LOSE_MARGIN:
        return "hc_wins"
    return "comparable"


# --- Sweeps und Tabellen -----------------------------------------------------------------------------------------------------------------------


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def run_config(base, seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS, **changes):
    """Mittel über die festen Instanzen und je `chains` Ketten-Seeds für die Einstellungen `base` mit `changes`."""
    s0 = replace(base, **changes)
    rows = []
    for seed in seeds:
        for ch in range(chains):
            a = analyse(replace(s0, seed=seed, chain_seed=ch))
            rows.append({"gap": a.gap, "other": a.other_gap, "single": a.single_gap, "starts": a.run_starts,
                         "other_starts": a.other_starts, "seconds": a.seconds, "crossings": a.crossings_end,
                         "single_iterations": a.single_iterations})
    out = {k: _mean(rows, k) for k in rows[0]}
    out.update({"gap_sd": float(np.std([r["gap"] for r in rows])), "gap_min": float(np.min([r["gap"] for r in rows])),
                "gap_max": float(np.max([r["gap"] for r in rows])), "n_runs": len(rows)})
    return out


SWEEP_VALUES = {"budget": (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000),
                "n": (10, 20, 40, 60, 100, 150, 200), "cluster_share": (0, 25, 50, 75, 100)}
SWEEP_LABELS = {"budget": "Budget (bewertete Nachbarn)", "n": "Stopps", "cluster_share": "Anteil in Gruppen (%)"}


def sweep(param, base=Settings(), values=None):
    values = SWEEP_VALUES[param] if values is None else values
    return [{"value": v, **run_config(base, **{param: v})} for v in values]


def single_descent_comparison(base=Settings(), seeds=C.SWEEP_SEEDS, chains=C.SWEEP_CHAINS):
    """Ein einzelner Abstieg (kein Neustart) je Modus, gleiche Startlösung: Bewertungen bis Konvergenz,
    Rescans/Züge, Endlänge - die "wie effizient ist EIN Abstieg" Kennzahl, unabhängig vom Budget-Sweep."""
    rows = {"dynasearch": [], "sequential": []}
    for seed in seeds:
        inst, D = instance(base.n, base.cluster_share, seed)
        bound = reference_bound(base.n, base.cluster_share, seed)
        for ch in range(chains):
            start = T.random_tour(len(D), np.random.default_rng(ch))
            rd = DA.dynasearch_descend(D, start, keep_steps=False)
            rs = T.descend(D, start, "2opt", "best", keep_steps=False)
            rows["dynasearch"].append({"evaluations": rd.evaluations, "iterations": rd.n_iterations, "moves": rd.n_moves,
                                        "gap": 100 * (rd.length - bound) / bound})
            rows["sequential"].append({"evaluations": rs.evaluations, "iterations": rs.n_moves,
                                        "gap": 100 * (rs.length - bound) / bound})
    return {mode: {k: float(np.mean([r[k] for r in vals])) for k in vals[0]} for mode, vals in rows.items()}


SCALING_N = C.SCALING_N
SCALING_POLICIES = (("Budget 200 Tausend", lambda n: 200000), ("Budget 5 000 · Stopps", lambda n: 5000 * n))


def scaling_table(base=Settings()):
    return [{"label": label, "rows": [{"value": n, **run_config(base, n=n, budget=fn(n))} for n in SCALING_N]} for label, fn in SCALING_POLICIES]


def chain_spread(settings, k=C.SPREAD_CHAINS):
    """k Ketten-Seeds auf derselben Instanz: der gewählte Modus gegen den jeweils anderen, gleiches Budget."""
    gaps, other_gaps = [], []
    for ch in range(k):
        a = analyse(replace(settings, chain_seed=ch))
        gaps.append(a.gap)
        other_gaps.append(a.other_gap)
    return {"run": np.array(gaps), "other": np.array(other_gaps)}
