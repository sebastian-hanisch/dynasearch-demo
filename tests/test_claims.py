"""Jede Zahl in den Hilfetexten, Presets, Tabellen und Grenzen der App ist hier über die fünf festen Sweep-Instanzen
(je drei Ketten-Seeds) belegt, mit denselben Auswertungsfunktionen wie die App selbst (`ev.run_config`/
`ev.single_descent_comparison`) - NIE über ein Ad-hoc-Skript mit abweichender Zufalls-Bindung. Diese Lehre stammt
aus der lin-kernighan-demo dieser Linie: dort benutzte eine Ad-hoc-Messung unabhängig gezogene rng-Instanzen statt
des echten paarigen start+seed-Musters von `analyse()`/`chained_run()` und erzeugte einen überzeugend falschen
Befund, der erst durch genau solche gegen die echte Implementierung geprüften Tests aufflog. Positive UND negative
Aussagen: Dynasearch gewinnt den Budget-Vergleich bei JEDEM Budget (positiv) - aber bei festem Budget verliert es
bei großen Instanzen (ab 100 Stopps) klar gegen sequentiell (negativ, ehrlicher Befund)."""

from functools import lru_cache

import pytest

import dyna_constants as C
import dyna_evaluation as ev


@lru_cache(maxsize=None)
def _cfg(items):
    return ev.run_config(ev.Settings(), **dict(items))


def cfg(**kw):
    return _cfg(tuple(sorted(kw.items())))


@lru_cache(maxsize=None)
def _single(n):
    return ev.single_descent_comparison(ev.Settings(n=n))


def near(value, expected, tol):
    assert abs(value - expected) <= tol, f"{value:.3f} statt {expected}"


# --- Budget-Sweep: Dynasearch gewinnt bei JEDEM gemessenen Budget (positiver Kernbefund) ------------------------------------------------------


@pytest.mark.parametrize("budget,dyna,seq,tol", [
    (10000, 5.81, 7.85, 0.5), (25000, 5.81, 7.85, 0.5), (50000, 5.81, 7.85, 0.5), (100000, 4.67, 7.85, 0.5),
    (200000, 3.75, 6.33, 0.5), (500000, 2.69, 3.98, 0.4), (1000000, 1.93, 2.58, 0.4), (2000000, 1.25, 1.91, 0.3),
])
def test_budget_sweep_numbers(budget, dyna, seq, tol):
    row = cfg(budget=budget)
    near(row["gap"], dyna, tol)
    near(row["other"], seq, tol)


def test_dynasearch_beats_sequential_at_every_measured_budget():
    for budget in C.BUDGETS:
        row = cfg(budget=budget)
        assert row["gap"] <= row["other"] + 1e-6                          # nie schlechter, meist klar besser


def test_standard_budget_matches_the_default_preset_help_text():
    std = cfg(budget=C.DEFAULT_BUDGET)
    near(std["gap"], 3.75, 0.5)
    near(std["other"], 6.33, 0.5)


# --- Ein einzelner Abstieg: Dynasearch braucht IMMER weniger Bewertungen, der Faktor waechst mit n ---------------------------------------------


@pytest.mark.parametrize("n,dyna_ev,seq_ev,tol", [(20, 2961, 3301, 400), (60, 50122, 101776, 8000), (100, 182123, 517665, 30000), (200, 1126283, 4438804, 150000)])
def test_single_descent_evaluation_numbers(n, dyna_ev, seq_ev, tol):
    r = _single(n)
    near(r["dynasearch"]["evaluations"], dyna_ev, tol)
    near(r["sequential"]["evaluations"], seq_ev, tol)


def test_dynasearch_always_needs_fewer_evaluations_and_the_ratio_grows_with_n():
    ratios = []
    for n in (20, 60, 100, 200):
        r = _single(n)
        assert r["dynasearch"]["evaluations"] < r["sequential"]["evaluations"]
        ratios.append(r["sequential"]["evaluations"] / r["dynasearch"]["evaluations"])
    assert ratios == sorted(ratios)                                       # der Faktor waechst monoton mit n
    assert ratios[-1] > 3.0                                               # bei n=200 rund 3.9x weniger


def test_moves_combined_per_dynasearch_iteration_grows_with_n():
    m20 = _single(20)["dynasearch"]["moves"] / _single(20)["dynasearch"]["iterations"]
    m60 = _single(60)["dynasearch"]["moves"] / _single(60)["dynasearch"]["iterations"]
    m200 = _single(200)["dynasearch"]["moves"] / _single(200)["dynasearch"]["iterations"]
    near(m20, 2.0, 0.5)
    near(m60, 4.0, 0.6)
    near(m200, 8.9, 1.0)
    assert m20 < m60 < m200


# --- Ehrlicher Befund: die GUETE eines einzelnen Abstiegs dreht sich mit n um (negativer Teilbefund) ----------------------------------------------


def test_single_descent_quality_crossover_with_instance_size():
    small = _single(20)
    assert small["dynasearch"]["gap"] < small["sequential"]["gap"] - 0.3   # Dynasearch klar besser bei kleinen Instanzen
    mid = _single(60)
    assert mid["dynasearch"]["gap"] < mid["sequential"]["gap"] - 0.3
    large = _single(200)
    assert large["dynasearch"]["gap"] > large["sequential"]["gap"] + 0.1   # sequentiell klar genauer bei großen Instanzen


# --- Skalierung bei festem Budget: der Budget-Vorteil kehrt sich ab ~100 Stopps um (negativer Kernbefund) ------------------------------------------


@pytest.mark.parametrize("n,dyna,seq,tol", [(20, 0.05, 0.05, 0.2), (40, 1.02, 1.04, 0.3), (60, 3.75, 6.33, 0.5),
                                             (100, 9.74, 8.78, 0.6), (150, 10.10, 9.49, 0.6), (200, 10.03, 9.56, 0.6)])
def test_scaling_numbers_at_two_hundred_thousand(n, dyna, seq, tol):
    row = ev.run_config(ev.Settings(), n=n, budget=200000)
    near(row["gap"], dyna, tol)
    near(row["other"], seq, tol)


def test_dynasearch_wins_up_to_sixty_stops_and_loses_from_one_hundred_on_at_fixed_budget():
    for n in (20, 40, 60):
        row = ev.run_config(ev.Settings(), n=n, budget=200000)
        assert row["gap"] <= row["other"] + 0.05
    for n in (100, 150, 200):
        row = ev.run_config(ev.Settings(), n=n, budget=200000)
        assert row["gap"] > row["other"] + 0.05


def test_large_preset_is_an_honest_negative_finding():
    """Das Preset 'Große Instanz (200 Stopps)' zeigt bewusst einen Fall, in dem Dynasearch VERLIERT."""
    row = ev.run_config(ev.Settings(), n=200, budget=200000)
    near(row["gap"], 10.03, 0.6)
    near(row["other"], 9.56, 0.6)
    assert row["gap"] > row["other"]


# --- Sonstiges ------------------------------------------------------------------------------------------------------------------------------------


def test_preset_count_matches_the_readme():
    assert len(C.PRESETS) == 5


def test_bound_is_positive_and_below_a_naive_upper_estimate():
    inst, D = ev.instance(60, 0, C.DEFAULT_SEED)
    bound = ev.reference_bound(60, 0, C.DEFAULT_SEED)
    assert 0 < bound < D.sum()                                           # grobe Plausibilität, keine scharfe Schranke
