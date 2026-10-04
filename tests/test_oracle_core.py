"""Orakel-Test für den Hill-Climbing-Kern (`dyna_tour`): Nachbarschafts-Deltas, Auswahlregeln, Bewertungszähler, Kreuzungszählung und 1-Baum-Schranke
gegen einen ANDEREN Rechenweg - Tourlängen werden hier durch vollständiges Neuberechnen nach dem Zerschneiden/Umbauen der Tourliste bestimmt
(nicht über die Delta-Formeln der Matrizen), Kreuzungen exakt mit Brüchen, die Schranke gegen die exakte Held-Karp-DP (kein Solver nötig).
Kein Test vergleicht den Code mit sich selbst oder mit eingefrorenen Werten."""

import itertools
from fractions import Fraction

import numpy as np

import dyna_tour as T

EPS = 1e-9


def _len(tour, Dl):
    n = len(tour)
    return sum(Dl[tour[k]][tour[(k + 1) % n]] for k in range(n))


def _neighbors(t, Dl, neighborhood):
    """(Art, neue Tour) aller Nachbarn in der dokumentierten Reihenfolge (swap/2opt: nach (i, j); or-opt: Länge, Stück i, Kante m der Tour)."""
    t = [int(x) for x in t]
    n = len(t)
    out = []
    if neighborhood == "swap":
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                u = t[:]
                u[i], u[j] = u[j], u[i]
                out.append(("swap", u))
    if neighborhood in ("2opt", "2opt+oropt"):
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                out.append(("2opt", t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]))
    if neighborhood in ("oropt", "2opt+oropt"):
        for length in (1, 2, 3):
            if n < length + 3:
                continue
            for i in range(n):
                seg = [t[(i + k) % n] for k in range(length)]
                rest = [t[(i + length + k) % n] for k in range(n - length)]
                for q in sorted(range(len(rest) - 1), key=lambda q: (i + length + q) % n):
                    best = min(((_len(rest[:q + 1] + s + rest[q + 1:], Dl), rest[:q + 1] + s + rest[q + 1:]) for s in (seg, seg[::-1])), key=lambda x: x[0])
                    out.append((f"oropt{length}", best[1]))
    return out


def _instances(rng, trials, n_lo, n_hi):
    for _ in range(trials):
        n = int(rng.integers(n_lo, n_hi))
        # Zufallspunkte und Gitterpunkte (viele Gleichstände, doppelte Punkte)
        xy = rng.random((n, 2)) * 100 if rng.random() < 0.6 else rng.integers(0, 4, (n, 2)).astype(float)
        yield n, xy, T.dist_matrix(xy)


def test_neighborhood_deltas_selection_rules_and_evaluation_counts_match_a_full_recomputation_oracle():
    rng = np.random.default_rng(2024)
    for n, xy, D in _instances(rng, 45, 4, 10):
        Dl = D.tolist()
        t = rng.permutation(n)
        base = T.tour_length(t, D)
        for neighborhood in T.NEIGHBORHOODS:
            orc = _neighbors(t, Dl, neighborhood)
            od = sorted(_len(u, Dl) - base for _, u in orc)
            assert np.allclose(od, sorted(T.neighbor_deltas(t, D, neighborhood).tolist()), atol=1e-7), (neighborhood, n)
            move, count = T.find_move(t, D, neighborhood, "best")
            assert count == len(orc)
            if od and od[0] < -EPS:
                assert move is not None and abs(move[-1] - od[0]) < 1e-7
                new = T.apply_move(t, move)
                assert abs(_len(new.tolist(), Dl) - (base + move[-1])) < 1e-7
            else:
                assert move is None
            move, count = T.find_move(t, D, neighborhood, "first")
            first = next((k for k, (_, u) in enumerate(orc) if _len(u, Dl) - base < -EPS), None)
            if first is None:
                assert move is None and count == len(orc)
            else:
                assert move is not None and count == first + 1
                assert abs(move[-1] - (_len(orc[first][1], Dl) - base)) < 1e-7
                assert abs(_len(T.apply_move(t, move).tolist(), Dl) - (base + move[-1])) < 1e-7


def test_descent_ends_in_a_local_optimum_found_by_full_recomputation():
    rng = np.random.default_rng(7)
    for n, xy, D in _instances(rng, 12, 5, 10):
        Dl = D.tolist()
        for neighborhood in T.NEIGHBORHOODS:
            for rule in T.RULES:
                r = T.descend(D, rng.permutation(n), neighborhood, rule)
                length = _len(r.tour.tolist(), Dl)
                assert abs(r.length - length) < 1e-7
                assert all(_len(u, Dl) >= length - 1e-7 for _, u in _neighbors(r.tour, Dl, neighborhood))


def _proper_crossing(p, q, r, s):
    """Echtes Schneiden zweier Strecken: p + a (q-p) = r + b (s-r) mit 0 < a, b < 1, exakt in Brüchen."""
    px, py = map(Fraction, p)
    qx, qy = map(Fraction, q)
    rx, ry = map(Fraction, r)
    sx, sy = map(Fraction, s)
    d1x, d1y, d2x, d2y = qx - px, qy - py, sx - rx, sy - ry
    den = d1x * d2y - d1y * d2x
    if den == 0:
        return False
    a = ((rx - px) * d2y - (ry - py) * d2x) / den
    b = ((rx - px) * d1y - (ry - py) * d1x) / den
    return 0 < a < 1 and 0 < b < 1


def test_count_crossings_matches_an_exact_fraction_oracle():
    rng = np.random.default_rng(11)
    for _ in range(150):
        n = int(rng.integers(4, 12))
        xy = rng.integers(0, 6, (n, 2)).astype(float) if rng.random() < 0.6 else rng.random((n, 2))
        t = rng.permutation(n)
        expected = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b, c, d = t[i], t[(i + 1) % n], t[j], t[(j + 1) % n]
                if len({a, b, c, d}) == 4 and _proper_crossing(xy[a], xy[b], xy[c], xy[d]):
                    expected += 1
        assert T.count_crossings(xy, t) == expected
    bowtie = np.array([[0, 0], [1, 1], [1, 0], [0, 1]], float)
    assert T.count_crossings(bowtie, [0, 1, 2, 3]) == 1 and T.count_crossings(bowtie, [0, 2, 1, 3]) == 0


def _held_karp_optimum(Dl):
    """Exakte kürzeste Rundtour (Held-Karp-DP über Teilmengen), n <= 9."""
    n = len(Dl)
    if n == 1:
        return 0.0
    if n == 2:
        return 2 * Dl[0][1]
    m = n - 1
    dp = {(1 << j, j): Dl[0][j + 1] for j in range(m)}
    for mask in range(1, 1 << m):
        for j in range(m):
            cur = dp.get((mask, j))
            if cur is None:
                continue
            for k in range(m):
                if not (mask >> k) & 1:
                    key = (mask | 1 << k, k)
                    v = cur + Dl[j + 1][k + 1]
                    if v < dp.get(key, float("inf")):
                        dp[key] = v
    return min(dp[((1 << m) - 1, j)] + Dl[j + 1][0] for j in range(m))


def test_dynamic_programming_oracle_agrees_with_brute_force():
    rng = np.random.default_rng(3)
    for _ in range(10):
        n = int(rng.integers(3, 8))
        Dl = T.dist_matrix(rng.random((n, 2)) * 50).tolist()
        brute = min(_len([0, *p], Dl) for p in itertools.permutations(range(1, n)))
        assert abs(_held_karp_optimum(Dl) - brute) < 1e-9


def test_one_tree_bound_never_exceeds_the_exact_optimum_and_degenerate_sizes_are_handled():
    rng = np.random.default_rng(5)
    for n, xy, D in _instances(rng, 40, 1, 9):
        Dl = D.tolist()
        opt = _held_karp_optimum(Dl)
        upper = T.descend(D, np.arange(n), "2opt+oropt", "best", keep_steps=False).length if n > 3 else opt
        assert T.held_karp_bound(D, upper, 300) <= opt + 1e-7


# --- Dynasearch-DP (dyna_algorithm): vollständige Enumeration unabhängiger Zugmengen mit Längen durch Neuberechnen ------------------------------


import dyna_algorithm as DA


def _two_opt_tour(t, i, j):
    return t[:i + 1] + t[i + 1:j + 1][::-1] + t[j + 1:]


def _candidates_by_recomputation(t, Dl, only_improving=True):
    """Alle 2-opt-Züge (i, j) ohne Position 0 und n-1 mit Gewinn aus vollständigem Neuberechnen der Tourlänge (nur die verbessernden, falls gewünscht)."""
    n = len(t)
    base = _len(t, Dl)
    out = []
    for i in range(1, n):
        for j in range(i + 2, n - 1):
            gain = base - _len(_two_opt_tour(t, i, j), Dl)
            if gain > EPS or not only_improving:
                out.append((i, j, gain))
    return out


def test_dynasearch_step_gain_equals_the_best_independent_subset_found_by_full_enumeration():
    rng = np.random.default_rng(31)
    checked = 0
    for n, xy, D in _instances(rng, 220, 6, 12):
        Dl = D.tolist()
        t = [int(x) for x in rng.permutation(n)]
        cands = _candidates_by_recomputation(t, Dl)[:14]
        full = _candidates_by_recomputation(t, Dl)
        if len(full) > 14:                                  # Enumeration nur bei überschaubarer Kandidatenzahl; sonst auch DP prüfen, aber nur gegen Schranken
            r = DA.dynasearch_step(t, D)
            assert r.gain >= max(g for _, _, g in full) - 1e-7
            continue
        best = 0.0
        for k in range(len(cands) + 1):
            for combo in itertools.combinations(cands, k):
                ordered = sorted(combo)
                if all(b[0] >= a[1] + 2 for a, b in zip(ordered, ordered[1:])):          # Fußabdrücke [i, j+1] disjunkt
                    best = max(best, sum(g for _, _, g in combo))
        r = DA.dynasearch_step(t, D)
        assert abs(r.gain - best) < 1e-7, (n, r.gain, best)
        assert abs((_len(t, Dl) - _len(r.tour.tolist(), Dl)) - r.gain) < 1e-7           # angewandt = tatsächliche Längenänderung
        assert sorted(r.tour.tolist()) == list(range(n))
        checked += 1
    assert checked >= 150


def test_touching_moves_are_additive_so_the_conservative_independence_rule_is_sufficient_but_not_necessary():
    """i2 = j1 + 1 (Fußabdrücke teilen nur den Knoten j1+1): Gewinne addieren sich; bei i2 <= j1 gilt das nicht allgemein."""
    rng = np.random.default_rng(41)
    touching = overlapping_nonadditive = 0
    for n, xy, D in _instances(rng, 150, 10, 15):
        Dl = D.tolist()
        t = [int(x) for x in rng.permutation(n)]
        base = _len(t, Dl)
        moves = _candidates_by_recomputation(t, Dl, only_improving=False)          # Additivität gilt für JEDES Paar, nicht nur für verbessernde Züge
        for (i1, j1, g1) in moves:
            for (i2, j2, g2) in moves:
                if i2 == j1 + 1:
                    new = _two_opt_tour(_two_opt_tour(t, i1, j1), i2, j2)               # gleiche Positionen: zweiter Zug liegt rechts vom ersten
                    assert abs((base - _len(new, Dl)) - (g1 + g2)) < 1e-7
                    touching += 1
                elif i1 < i2 <= j1 < j2 or (i1 < i2 and j2 < j1):
                    new = _two_opt_tour(_two_opt_tour(t, i1, j1), i2, j2)
                    if abs((base - _len(new, Dl)) - (g1 + g2)) > 1e-7:
                        overlapping_nonadditive += 1
    assert touching > 2000 and overlapping_nonadditive > 2000
