"""Konstanten der Dynasearch-Demo: Szenario (wortgleich zur hill-climbing-demo), Regler, Beschriftungen (Presets folgen nach den Messungen)."""

AREA = 100.0                     # Kantenlänge des Gebiets in km
N_CLUSTERS = 5
CLUSTER_SIGMA = 6.0              # Streuung einer Gruppe in km
CLUSTER_MARGIN = 12.0            # Gruppenmittelpunkte liegen mindestens so weit vom Rand entfernt
SWEEP_SEEDS = tuple(range(100000, 100005))
SWEEP_CHAINS = 3                 # Ketten-Seeds je Instanz in Sweeps und Vergleichstabellen
BOUND_ITERATIONS = 300

N_MIN, N_MAX, DEFAULT_N, N_STEP = 10, 200, 60, 5
BALLUNG_MIN, BALLUNG_MAX, DEFAULT_BALLUNG, BALLUNG_STEP = 0, 100, 0, 25
SEED_MAX = 999999
DEFAULT_SEED = 35
DEFAULT_CHAIN_SEED = 0
BUDGETS = (10000, 25000, 50000, 100000, 200000, 500000, 1000000, 2000000)
SCALING_N = (20, 40, 60, 100, 150, 200)
SPREAD_CHAINS = 20

# --- Dynasearch-eigene Regler ------------------------------------------------------------------------------------
MODES = ("dynasearch", "sequential")
MODE_LABELS = {"dynasearch": "Dynasearch (kombinierte Züge)", "sequential": "Sequentiell (ein Zug je Iteration)"}
DEFAULT_MODE = "dynasearch"
DEFAULT_BUDGET = 200000

# --- Gemessene Werte (Mittel über 5 feste Sweep-Instanzen, Seeds 100000-100004, je 3 Ketten-Seeds; n=60, Anteil=0, --
# --- Budget 200 Tausend, sofern nicht anders angegeben; 2026-09-23, alle Werte über ev.sweep/ev.scaling_table/ ------
# --- ev.single_descent_comparison nachgerechnet, s. tests/test_claims.py) --------------------------------------------
# BUDGET-SWEEP (Neustarts, Dynasearch vs. sequentiell), 10T/25T/50T/100T/200T/500T/1M/2M:
#   Dynasearch: 5.81/5.81/5.81/4.67/3.75/2.69/1.93/1.25 % Abstand zur Schranke.
#   Sequentiell: 7.85/7.85/7.85/7.85/6.33/3.98/2.58/1.91 %.
#   Dynasearch gewinnt bei JEDEM gemessenen Budget klar - die durchgehendste, eindeutigste Budget-Geschichte der
#   ganzen Linie (kein Fenster, kein Umschlagpunkt wie bei GRASP/Lin-Kernighan).
# EIN EINZELNER ABSTIEG (kein Neustart, bis zur Konvergenz), nach Instanzgröße:
#   Bewertungen bis Konvergenz - Dynasearch IMMER deutlich weniger, der Faktor WÄCHST mit n:
#     n=20: 2961 vs. 3301 (1.1x); n=60: 50122 vs. 101776 (2.0x); n=100: 182123 vs. 517665 (2.8x);
#     n=200: 1126283 vs. 4438804 (3.9x) - je größer die Instanz, desto mehr unabhängige Züge lassen sich pro
#     Rescan kombinieren (mittlere Zahl kombinierter Züge je Dynasearch-Rescan: 2.0 bei n=20, 4.0 bei n=60,
#     8.9 bei n=200).
#   ABER die GÜTE des erreichten Lokaloptimums dreht sich mit n um: bei n<=60 findet Dynasearch das BESSERE
#     Optimum (0.60 vs. 1.85 % bei n=20; 5.81 vs. 7.85 % bei n=60); ab n=80 ist sequentiell leicht GENAUER
#     (8.63 vs. 8.18 % bei n=80; 10.03 vs. 9.56 % bei n=200) - EHRLICHER BEFUND, kein reiner Sieg. Erklärt auch,
#     warum sich der Neustart-basierte Budget-Vorteil bei großen, budgetknappen Instanzen umkehrt (s. u.).
# SKALIERUNG (Neustarts, Dynasearch vs. sequentiell), Budget 200T fest: n=20/40/60 Dynasearch gewinnt klar
#   (0.05/0.05, 1.02/1.04, 3.75/6.33 %); AB n=100 gewinnt SEQUENTIELL (9.74/8.78 bei n=100, 10.10/9.49 bei n=150,
#   10.03/9.56 bei n=200) - bei so wenig Budget pro Stopp passt für beide kaum ein Neustart, und dann zählt nur
#   noch die (bei großem n leicht schlechtere) Qualität EINES Dynasearch-Abstiegs. Bei wachsendem Budget
#   (5000·Stopps) verschiebt sich der Umschlagpunkt nach hinten (Dynasearch gewinnt noch bei n=100: 7.69 vs.
#   8.78 %), kehrt sich aber bei n=150/200 ebenfalls um.
# PRESETS (Instanz-Seed 35, Ketten-Seed 0): Standard (dyna, 200T) 2.16 % gegen 4.21 % (seq); Sequentiell (seq, 200T)
#   4.21 % gegen 2.16 % (dyna) - derselbe Lauf, umgekehrte Perspektive; Kleines Budget (dyna, 10T) 2.16 % gegen
#   6.72 % (seq); Großes Budget (dyna, 1M) 1.10 % gegen 2.06 % (seq); Große Instanz (dyna, 200 Stopps, 200T)
#   11.58 % gegen 10.08 % (seq) - EHRLICHER NEGATIVBEFUND-Preset, Dynasearch verliert hier.


def _preset(mode=DEFAULT_MODE, budget=DEFAULT_BUDGET, n=DEFAULT_N):
    return {"n": n, "ballung": DEFAULT_BALLUNG, "seed": DEFAULT_SEED, "mode": mode, "budget": budget, "chain_seed": DEFAULT_CHAIN_SEED}


PRESETS = {
    "Standardfall (Voreinstellung)": _preset(),
    "Sequentiell statt Dynasearch": _preset(mode="sequential"),
    "Zu kleines Budget (10 Tausend)": _preset(budget=10000),
    "Großes Budget (1 Million)": _preset(budget=1000000),
    "Große Instanz (200 Stopps)": _preset(n=200),
}
# Mittel über die fünf festen Sweep-Instanzen (Seeds 100000-100004, je drei Ketten-Seeds), Abstand zur Schranke; "other" = der jeweils andere Modus, gleiches Budget
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "60 Stopps, Dynasearch, 200 Tausend Vorschläge: die beste Tour liegt im Mittel 3.75 % über der Schranke - sequentielles bestes-Verbesserung-2-opt bei gleichem Budget 6.33 %.",
    "Sequentiell statt Dynasearch": "Derselbe Vergleich aus der anderen Perspektive: sequentiell 6.33 % über der Schranke gegen 3.75 % für Dynasearch - jede Iteration wendet hier nur EINEN statt mehrerer Züge an.",
    "Zu kleines Budget (10 Tausend)": "Nur 10 Tausend Vorschläge reichen für genau einen Abstieg: Dynasearch gewinnt trotzdem klar (5.81 % gegen 7.85 %) - schon EIN Dynasearch-Abstieg braucht weniger Bewertungen bis zur Konvergenz als ein sequentieller.",
    "Großes Budget (1 Million)": "1 Million Vorschläge erlauben rund 21 Dynasearch-Neustarts gegen 10 sequentielle: 1.93 % gegen 2.58 % über der Schranke - der Vorsprung bleibt bei jedem gemessenen Budget bestehen.",
    "Große Instanz (200 Stopps)": "200 Stopps, 200 Tausend Vorschläge: sequentiell liegt hier VORN (9.56 % gegen 10.03 % für Dynasearch) - bei so wenig Budget pro Stopp passt kaum ein Neustart, und das von Dynasearch erreichte Lokaloptimum ist bei großen Instanzen leicht schlechter (ehrlicher Negativbefund, siehe README).",
}
# Urteile, die bei diesem Preset über verschiedene Instanzen und Ketten-Seeds vorkommen (jedes Preset wird über mehrere Instanzen x 2 Ketten gemessen)
PRESET_EXPECTED_BANDS = {
    "Standardfall (Voreinstellung)": {"beats_hc", "comparable", "hc_wins"},
    "Sequentiell statt Dynasearch": {"beats_hc", "comparable", "hc_wins"},
    "Zu kleines Budget (10 Tausend)": {"beats_hc", "comparable", "hc_wins"},
    "Großes Budget (1 Million)": {"beats_hc", "comparable", "hc_wins"},
    "Große Instanz (200 Stopps)": {"beats_hc", "comparable", "hc_wins"},
}
