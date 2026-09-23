"""Dynasearch (Potts & van de Velde 1995; Congram, Potts & van de Velde 2002) für eine Rundtour (TSP): statt wie ein
gewöhnlicher 2-opt-Abstieg (beste Verbesserung, voller Rescan) pro Iteration GENAU EINEN verbessernden Zug
anzuwenden, sucht Dynasearch die gewinn-maximale MENGE paarweise UNABHÄNGIGER (nicht überlappender) verbessernder
2-opt-Züge und wendet sie gleichzeitig an. "Unabhängig" heißt: die beiden umgekehrten Segmente [i₁+1,j₁] und
[i₂+1,j₂] überlappen sich nicht - dann ändert der eine Zug keine der Kanten, von denen der Delta-Wert des anderen
abhängt (dieselbe Delta-Formel wie ein einzelner 2-opt-Zug bleibt gültig, die Gewinne sind additiv, unabhängig von
der Reihenfolge). Diese Menge zu finden ist im Allgemeinen eine Wahl aus exponentiell vielen Teilmengen - Dynasearch
löst das exakt in O(n²) durch dynamische Programmierung (klassisches gewichtetes Intervall-Scheduling: jeder
verbessernde Zug ist ein Intervall [i+1,j] mit Gewinn -delta[i,j], gesucht ist die gewinn-maximale Menge paarweise
disjunkter Intervalle).

Kandidaten-Erzeugung nutzt `dyna_tour._delta_2opt`/`_valid_pairs` unverändert (kein eigenständig hergeleitetes
Delta-Risiko - dieselbe vektorisierte Matrix wie beim sequentiellen Abstieg aus der Hill-Climbing-Demo).

Zentrale, beweisbare Eigenschaft: konvergiert Dynasearch (keine Kombination mit positivem Gesamtgewinn mehr), ist
die Tour IMMER ein echtes 2-opt-Lokaloptimum - jeder einzelne verbessernde 2-opt-Zug wäre selbst eine gültige
1-Intervall-"Kombination" und würde von der DP gefunden (siehe tests/test_algorithm.py, geprüft gegen
`dyna_tour.is_local_optimum`, unabhängig von der eigenen DP-Logik)."""

from dataclasses import dataclass, field

import numpy as np

import dyna_tour as T

EPS = 1e-9


@dataclass
class StepResult:
    tour: np.ndarray
    gain: float                # Gesamtgewinn dieses Schritts (>= 0; 0 = keine Verbesserung mehr, konvergiert)
    evaluations: int           # Zahl der geprüften (i,j)-Paare (voller Rescan, wie ein sequentieller Zug)
    n_moves: int                # Zahl der GLEICHZEITIG angewandten, paarweise unabhängigen 2-opt-Züge


def dynasearch_step(t, D):
    """Eine DP-Iteration: sammelt alle gültigen, verbessernden 2-opt-Kandidaten aus der vollen Delta-Matrix, löst
    das gewichtete Intervall-Scheduling-Problem (maximale Gesamtsumme paarweise UNABHÄNGIGER Züge) mit einer
    einzigen Vorwärts-DP über die Tour-Positionen, wendet die gewählte Menge gleichzeitig an (Reihenfolge egal, da
    unabhängig - jede Umkehrung ist eine reine Array-Slice-Operation wie im sequentiellen Abstieg).

    Ein einzelner 2-opt-Zug (i,j) referenziert VIER Positionen, nicht nur das umgekehrte Stück [i+1,j]: die
    entfernten Kanten sind (t[i],t[i+1]) und (t[j],t[j+1]) - Position i UND j+1 liegen AUSSERHALB des umgekehrten
    Stücks, werden aber von der Delta-Formel gelesen. Zwei Züge sind deshalb nur unabhängig, wenn ihre VIER-Positionen-
    Fußabdrücke [i,j+1] sich nicht überlappen UND nicht berühren (sonst läse der zweite Zug eine Position, die der
    erste bereits verändert hat - siehe tests/test_algorithm.py für den Fehler, den eine zu lockere Bedingung
    (nur das umgekehrte Stück, nicht den vollen Fußabdruck) tatsächlich verursacht hat). Konkret: zwei Züge mit
    i1 < i2 sind unabhängig genau dann, wenn i2 >= j1 + 2. Wie in der Literatur (Potts & van de Velde 1995: "moves
    that do not involve city 1") werden Züge, die Position 0 (das Depot) referenzieren, deshalb komplett
    ausgeschlossen (i >= 1 UND j <= n-2, damit auch j+1 nie auf Position 0 zurückwickelt) - macht die Tour für die
    DP zu einer linearen Kette 0..n-1 statt eines Zyklus, ohne Sonderfall für den Wickel-Übergang."""
    t = np.asarray(t, dtype=np.int64)
    n = len(t)
    delta, ok = T._delta_2opt(t, D)
    evaluations = int(ok.sum())
    i_idx, j_idx = np.indices((n, n))
    dyna_ok = ok & (i_idx >= 1) & (j_idx <= n - 2)
    improving = dyna_ok & (delta < -EPS)
    ii, jj = np.where(improving)
    if len(ii) == 0:
        return StepResult(t.copy(), 0.0, evaluations, 0)
    gains = (-delta[ii, jj]).tolist()

    # Der VOLLE Fußabdruck eines Zugs (i,j) ist das GESCHLOSSENE Intervall [i, j+1] (4 referenzierte Positionen
    # i, i+1, j, j+1) - als HALB-OFFENES Intervall [i, j+2) geschrieben, damit sich Fußabdrücke wie gewohnte
    # Halbintervalle aneinanderreihen lassen. Gruppiert wird nach dem Fußabdruck-ENDE j+2 (nicht j+1 - das wäre
    # der Fehler von oben: ein Zug mit Fußabdruck [i,j+2) darf erst mit `prev[i]` kombiniert werden, also mit
    # Zügen, deren eigener Fußabdruck VOR Position i endet - nicht mit `prev[i-1]` oder einer um 1 verschobenen
    # Gruppierung, sonst überlappen sich die Fußabdrücke exakt an der Position, die beide Züge referenzieren).
    by_end = [[] for _ in range(n + 1)]
    for i, j, g in zip(ii.tolist(), jj.tolist(), gains):
        by_end[j + 2].append((i, j, g))

    prev = [0.0] * (n + 1)                    # prev[p] = bester Gesamtgewinn nur mit Zügen, deren Fußabdruck [.,p) vollständig vor p liegt
    pick = [None] * (n + 1)                   # pick[p] = (i, j, gain), falls ein bei Fußabdruck-Ende p gewählter Zug
    for p in range(n + 1):
        best_here = prev[p - 1] if p >= 1 else 0.0
        best_pick = None
        for i, j, gain in by_end[p]:
            cand_total = prev[i] + gain        # prev[i]: Fußabdrücke vor Position i frei, [i,j+2) durch diesen Zug belegt
            if cand_total > best_here:
                best_here = cand_total
                best_pick = (i, j, gain)
        prev[p] = best_here
        pick[p] = best_pick

    total_gain = prev[n]
    if total_gain <= EPS:
        return StepResult(t.copy(), 0.0, evaluations, 0)

    chosen = []
    p = n
    while p >= 0:
        if pick[p] is not None:
            i, j, gain = pick[p]
            chosen.append((i, j))
            p = i
        else:
            p -= 1
    chosen.reverse()

    new_t = t.copy()
    for i, j in chosen:
        new_t[i + 1:j + 1] = new_t[i + 1:j + 1][::-1]
    return StepResult(new_t, float(total_gain), evaluations, len(chosen))


@dataclass
class Descent:
    tour: np.ndarray
    length: float
    steps: list = field(default_factory=list)     # Start + ein Eintrag je Dynasearch-Iteration (bei keep_steps=False nur der Start)
    evaluations: int = 0
    n_moves: int = 0                                # Summe der einzelnen kombinierten 2-opt-Züge über alle Iterationen
    n_iterations: int = 0                           # Zahl der Dynasearch-Iterationen (volle Rescans)


def dynasearch_descend(D, tour, max_moves=100000, keep_steps=True, max_evaluations=None):
    """Wiederholt `dynasearch_step`, bis keine verbessernde Kombination mehr existiert, `max_moves` Iterationen
    oder das Bewertungsbudget `max_evaluations` erreicht sind. Da die DP Züge, die Position 0 (das Depot)
    referenzieren, bewusst ausschließt (s. `dynasearch_step`-Docstring), ist die Dynasearch-Konvergenz allein NUR
    ein 2-opt-Lokaloptimum unter Ausschluss depot-berührender Züge - ein einzelner klassischer sequentieller
    2-opt-Aufräum-Durchgang danach (`dyna_tour.descend`, volle Nachbarschaft inkl. Depot) schließt diese Lücke und
    garantiert ein ECHTES 2-opt-Lokaloptimum (getestet gegen `dyna_tour.is_local_optimum`, unabhängig von der
    eigenen DP-Logik)."""
    t = np.asarray(tour, dtype=np.int64).copy()
    length = T.tour_length(t, D)
    steps = [T.Step(None, t.copy(), length)] if keep_steps else []
    evaluations = n_moves = n_iterations = 0
    for _ in range(max_moves):
        if max_evaluations is not None and evaluations >= max_evaluations:
            break
        result = dynasearch_step(t, D)
        evaluations += result.evaluations
        if result.n_moves == 0:
            break
        t = result.tour
        length -= result.gain
        n_moves += result.n_moves
        n_iterations += 1
        if keep_steps:
            steps.append(T.Step(("dynasearch", result.n_moves), t.copy(), length))

    remaining = None if max_evaluations is None else max(0, max_evaluations - evaluations)
    if remaining is None or remaining > 0:
        cleanup = T.descend(D, t, "2opt", "best", keep_steps=False, max_evaluations=remaining)
        evaluations += cleanup.evaluations
        if cleanup.n_moves > 0:
            t = cleanup.tour
            n_moves += cleanup.n_moves
            n_iterations += 1
            if keep_steps:
                steps.append(T.Step(("cleanup", cleanup.n_moves), t.copy(), cleanup.length))

    length = T.tour_length(t, D)                    # Rundungsfehler der Delta-Summen beseitigen
    return Descent(t, length, steps, evaluations, n_moves, n_iterations)
