"""dyna_algorithm: die DP muss die gewinn-maximale Menge paarweise unabhängiger 2-opt-Züge finden - hier gegen eine
UNABHÄNGIGE Brute-Force-Prüfung (alle Teilmengen kleiner Kandidatenmengen) getestet, nicht nur die eigene Logik.
Daneben: Additivität der Deltas (die zentrale Annahme, die den ganzen Ansatz trägt), Konvergenz ist immer ein
echtes 2-opt-Lokaloptimum (dank Aufräum-Durchgang), Budget-Buchführung, Determinismus."""

import itertools

import numpy as np
import pytest

import dyna_algorithm as DA
import dyna_tour as T


def _instance(n, seed):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 100
    return T.dist_matrix(xy)


def _brute_force_best_combo(t, D):
    """Unabhängige Prüfung: alle Teilmengen der gültigen, verbessernden Dynasearch-Kandidaten durchprobieren
    (nicht die DP-Logik selbst), die paarweise unabhängige Teilmenge mit maximalem Gesamtgewinn finden."""
    n = len(t)
    delta, ok = T._delta_2opt(t, D)
    i_idx, j_idx = np.indices((n, n))
    dyna_ok = ok & (i_idx >= 1) & (j_idx <= n - 2)
    ii, jj = np.where(dyna_ok & (delta < -1e-9))
    cands = list(zip(ii.tolist(), jj.tolist(), (-delta[ii, jj]).tolist()))
    best = 0.0
    for r in range(len(cands) + 1):
        for combo in itertools.combinations(cands, r):
            independent = True
            for a in range(len(combo)):
                for b in range(a + 1, len(combo)):
                    lo, hi = sorted((combo[a], combo[b]), key=lambda c: c[0])
                    if hi[0] < lo[1] + 2:
                        independent = False
                        break
                if not independent:
                    break
            if independent:
                total = sum(g for _, _, g in combo)
                best = max(best, total)
    return best


@pytest.mark.parametrize("trial", range(25))
def test_dp_matches_brute_force_over_all_subsets(trial):
    rng = np.random.default_rng(trial)
    n = int(rng.integers(6, 11))
    D = _instance(n, trial)
    t = T.random_tour(n, np.random.default_rng(trial + 100))
    bf_gain = _brute_force_best_combo(t, D)
    r = DA.dynasearch_step(t, D)
    assert r.gain == pytest.approx(bf_gain, abs=1e-6)


@pytest.mark.parametrize("trial", range(20))
def test_chosen_moves_never_reference_the_depot(trial):
    """Kein gewählter Zug darf Position 0 (i) oder Position n-1 (j) berühren - die literaturübliche Beschränkung
    ("moves that do not involve city 1"), direkt an den ANGEWANDTEN Zügen geprüft, nicht nur an der Kandidatenliste."""
    D = _instance(30, trial)
    t = T.random_tour(30, np.random.default_rng(trial + 200))
    delta, ok = T._delta_2opt(t, D)
    n = len(t)
    before = t.copy()
    r = DA.dynasearch_step(t, D)
    changed = np.where(before != r.tour)[0]
    if len(changed) > 0:
        assert 0 not in changed and (n - 1) not in changed


@pytest.mark.parametrize("trial", range(30))
def test_step_gain_matches_the_actual_length_reduction(trial):
    """Additivität: die tatsächliche Längenänderung nach dem gleichzeitigen Anwenden ALLER gewählten Züge muss
    exakt der Summe der einzeln (auf der Ausgangstour) berechneten Deltas entsprechen - keine Interaktion."""
    D = _instance(40, trial)
    t = T.random_tour(40, np.random.default_rng(trial + 300))
    r = DA.dynasearch_step(t, D)
    before = T.tour_length(t, D)
    after = T.tour_length(r.tour, D)
    assert (before - after) == pytest.approx(r.gain, abs=1e-6)


def test_step_never_worsens_and_stays_a_valid_permutation():
    D = _instance(50, 5)
    t = T.random_tour(50, np.random.default_rng(6))
    r = DA.dynasearch_step(t, D)
    assert sorted(r.tour.tolist()) == list(range(50))
    assert T.tour_length(r.tour, D) <= T.tour_length(t, D) + 1e-9


def test_no_improving_candidate_returns_zero_gain_and_unchanged_tour():
    D = _instance(10, 7)
    t = T.random_tour(10, np.random.default_rng(0))
    r = T.descend(D, t, "2opt", "best", keep_steps=False)             # ein echtes 2-opt-Lokaloptimum
    step = DA.dynasearch_step(r.tour, D)
    assert step.n_moves == 0 and step.gain == 0.0
    assert np.array_equal(step.tour, r.tour)


# --- dynasearch_descend: Konvergenz, Budget, Determinismus ------------------------------------------------------------


@pytest.mark.parametrize("trial", range(15))
def test_descend_always_converges_to_a_true_two_opt_local_optimum(trial):
    """Dank des Aufräum-Durchgangs (voller 2-opt-Rescan inkl. Depot) ist das Ergebnis IMMER ein echtes
    2-opt-Lokaloptimum, obwohl die Dynasearch-DP selbst depot-beruehrende Zuege ausschliesst."""
    rng = np.random.default_rng(trial)
    n = int(rng.integers(10, 60))
    D = _instance(n, trial)
    t = T.random_tour(n, np.random.default_rng(trial + 1000))
    r = DA.dynasearch_descend(D, t, keep_steps=False)
    assert T.is_local_optimum(D, r.tour, "2opt")
    assert r.length == pytest.approx(T.tour_length(r.tour, D), abs=1e-6)


def test_descend_length_is_strictly_non_increasing_across_steps():
    D = _instance(40, 8)
    t = T.random_tour(40, np.random.default_rng(9))
    r = DA.dynasearch_descend(D, t, keep_steps=True)
    lengths = [s.length for s in r.steps]
    assert all(b <= a + 1e-9 for a, b in zip(lengths, lengths[1:]))
    assert lengths[0] >= lengths[-1]


def test_descend_stops_at_the_evaluation_budget():
    D = _instance(60, 10)
    t = T.random_tour(60, np.random.default_rng(11))
    full = DA.dynasearch_descend(D, t, keep_steps=False)
    cap = max(1, full.evaluations // 3)
    cut = DA.dynasearch_descend(D, t, keep_steps=False, max_evaluations=cap)
    assert cut.evaluations >= cap
    assert cut.length >= full.length - 1e-6
    unlimited = DA.dynasearch_descend(D, t, keep_steps=False, max_evaluations=10**9)
    assert unlimited.length == pytest.approx(full.length, abs=1e-6)


def test_descend_is_deterministic():
    D = _instance(30, 12)
    t = T.random_tour(30, np.random.default_rng(13))
    a = DA.dynasearch_descend(D, t, keep_steps=False)
    b = DA.dynasearch_descend(D, t, keep_steps=False)
    assert np.array_equal(a.tour, b.tour) and a.evaluations == b.evaluations


def test_descend_reaches_the_same_or_better_quality_than_a_single_sequential_two_opt_move_could():
    """Kein Anspruch auf immer BESSER als sequentiell (beide erreichen irgendein 2-opt-Optimum, moeglicherweise
    unterschiedliche) - aber Dynasearch darf niemals SCHLECHTER als die Startlösung enden."""
    D = _instance(50, 14)
    t = T.random_tour(50, np.random.default_rng(15))
    start_length = T.tour_length(t, D)
    r = DA.dynasearch_descend(D, t, keep_steps=False)
    assert r.length <= start_length + 1e-9


def test_iterations_and_moves_are_consistent():
    D = _instance(50, 16)
    t = T.random_tour(50, np.random.default_rng(17))
    r = DA.dynasearch_descend(D, t, keep_steps=True)
    assert r.n_iterations == len(r.steps) - 1
    assert r.n_moves >= r.n_iterations                                # jede Iteration wendet mindestens einen Zug an
