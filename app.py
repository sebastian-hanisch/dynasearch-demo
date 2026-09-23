"""Dynasearch - eine Lieferrunde, deren Suche mehrere unabhängige Züge auf einmal kombiniert - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Achtes Stück der Trajektorien-Metaheuristiken-Linie der "Konzepte"-Reihe, zweites Stück des Nachbarschafts-Zweigs
(Kind der Wurzel Hill Climbing, wie Lin-Kernighan). Ein normaler 2-opt-Abstieg wendet pro Iteration genau EINEN
verbessernden Zug an; Dynasearch (Potts & van de Velde 1995; Congram, Potts & van de Velde 2002) sucht stattdessen
die gewinn-maximale MENGE paarweise unabhängiger (nicht überlappender) verbessernder Züge und wendet sie alle auf
einmal an - nicht durch Ausprobieren aller exponentiell vielen Teilmengen, sondern exakt durch eine dynamische
Programmierung in O(n²). Siehe README für die Einordnung.

Lauffähig mit: streamlit run app.py
"""

import time
from dataclasses import replace

import numpy as np
import streamlit as st

import dyna_algorithm as DA
import dyna_constants as C
import dyna_evaluation as EV
import dyna_tour as T
from dyna_evaluation import SWEEP_LABELS, Settings, analyse, chain_spread, scaling_table, single_descent_comparison, sweep, verdict
from dyna_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_chain_seed,
    randomize_seed,
    sync_query_params,
)
from dyna_visualization import build_budget, build_instance, build_moves_per_iteration, build_scaling, build_spread, build_sweep, build_tour, build_trace

st.set_page_config(page_title="Dynasearch – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner=False)
def _run_with_steps(settings):
    inst, D = EV.instance(settings.n, settings.cluster_share, settings.seed)
    bound = EV.reference_bound(settings.n, settings.cluster_share, settings.seed)
    start = T.random_tour(len(D), np.random.default_rng(settings.chain_seed))
    # Ungedeckelt, wie der erste Neustart in ev.restarts() (läuft laut dortigem Docstring immer bis zur Konvergenz) -
    # damit zeigt die Schritt-Ansicht denselben Abstieg, der auch in die Ergebnis-Kennzahlen eingeht.
    if settings.mode == "dynasearch":
        run = DA.dynasearch_descend(D, start, keep_steps=True)
    else:
        run = T.descend(D, start, "2opt", "best", keep_steps=True)
    return inst, D, bound, run


@st.cache_data(show_spinner=False)
def _sweep(param, base):
    return sweep(param, base)


@st.cache_data(show_spinner=False)
def _single_descent(base):
    return single_descent_comparison(base)


@st.cache_data(show_spinner=False)
def _spread(base):
    return chain_spread(base)


@st.cache_data(show_spinner=False)
def _scaling(base):
    return scaling_table(base)


def _fmt_int(x):
    return f"{int(round(x)):,}".replace(",", ".")


def _changed_nodes(prev_tour, curr_tour):
    before, after = T.tour_edges(prev_tour), T.tour_edges(curr_tour)
    return {node for edge in (before ^ after) for node in edge}


st.title("🧩 Dynasearch – eine Suche, die mehrere unabhängige Züge auf einmal kombiniert")
st.markdown(
    """
Ein normaler 2-opt-Abstieg wendet pro Iteration **genau einen** verbessernden Zug an, dann wird die komplette
Nachbarschaft neu bewertet. **Dynasearch** (Potts & van de Velde 1995) tut mehr mit derselben vollen Bewertung:
es sucht die **gewinn-maximale Menge paarweise unabhängiger** (nicht überlappender) verbessernder Züge und wendet
sie **alle gleichzeitig** an - eine Auswahl aus exponentiell vielen möglichen Kombinationen, aber exakt gelöst
durch **dynamische Programmierung** in O(n²), nicht durch Ausprobieren. Braucht das weniger Bewertungen bis zur
Konvergenz als ein Zug nach dem anderen - und zahlt sich das am Ende auch in der Tourqualität aus?
"""
)
st.caption(
    "Achtes Stück der Trajektorien-Metaheuristiken-Linie der \"Konzepte\"-Reihe, zweites Stück des "
    "**Nachbarschafts-Zweigs** (Kind der Wurzel, wie Lin-Kernighan): dieselbe Rundtour wie in der "
    "[hill-climbing-demo](https://sebastianhanisch-hill-climbing-demo.streamlit.app/), der "
    "[simulated-annealing-demo](https://sebastianhanisch-simulated-annealing-demo.streamlit.app/), der "
    "[iterated-local-search-demo](https://github.com/sebastian-hanisch/iterated-local-search-demo), der "
    "[variable-neighborhood-search-demo](https://github.com/sebastian-hanisch/variable-neighborhood-search-demo), der "
    "[tabu-search-demo](https://github.com/sebastian-hanisch/tabu-search-demo), der "
    "[grasp-demo](https://github.com/sebastian-hanisch/grasp-demo) und der "
    "[lin-kernighan-demo](https://github.com/sebastian-hanisch/lin-kernighan-demo) - ein Depot in der Mitte, n "
    "Kundenstopps in einem 100 × 100-km-Gebiet, euklidische Entfernungen."
)

with st.expander("So funktioniert Dynasearch", expanded=True):
    st.markdown(
        """
1. **Kandidaten sammeln.** Wie beim vollen 2-opt-Rescan werden alle O(n²) möglichen Züge bewertet; nur die
   verbessernden werden behalten. Jeder Zug ist ein "Intervall" auf der Tour mit Gewinn = Längenersparnis.
2. **Unabhängigkeit.** Zwei Züge sind unabhängig, wenn ihre betroffenen Tour-Abschnitte sich nicht überlappen -
   dann ändert der eine Zug keine der Kanten, von denen der Delta-Wert des anderen abhängt (die Gewinne addieren
   sich exakt).
3. **Dynamische Programmierung.** Statt alle 2^k Teilmengen der k Kandidaten durchzuprobieren, findet eine DP in
   EINEM Durchgang die gewinn-maximale Menge paarweise unabhängiger Züge - ein klassisches gewichtetes
   Intervall-Scheduling-Problem, exakt lösbar in O(n²).
4. **Alle gleichzeitig anwenden.** Die gewählte Menge wird auf einmal angewandt (Reihenfolge egal, da unabhängig) -
   das kann mehrere weit auseinanderliegende Stellen der Tour auf einen Schlag verbessern.
5. **Depot ausgenommen.** Wie im Original-Paper referenziert kein Zug Position 0 (das Depot) - ein anschließender
   normaler 2-opt-Aufräum-Durchgang schließt diese Lücke, damit das Ergebnis immer ein echtes 2-opt-Lokaloptimum ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name], key=f"preset_{name}")

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_stops = st.slider(
        "Stopps", *bounds("n_slider"), key="n_slider", step=C.N_STEP,
        help="Anzahl der Kundenstopps (das Depot kommt dazu). Je größer die Instanz, desto mehr unabhängige Züge lassen sich je Dynasearch-Rescan kombinieren (im Mittel 2.0 bei 20 Stopps, 8.9 bei 200) - aber die erreichte Tourqualität ist dann nicht mehr automatisch besser als sequentiell (siehe Skalierungs-Experiment).",
    )
    cluster_share = st.slider(
        "Anteil der Stopps in Gruppen [%]", *bounds("ballung_slider"), key="ballung_slider", step=C.BALLUNG_STEP,
        help="Wie viele Stopps in fünf Gruppen (Städten) liegen statt gleichverteilt im Gebiet.",
    )
    mode = st.radio(
        "Modus", list(C.MODES), key="mode_radio", format_func=lambda m: C.MODE_LABELS[m], horizontal=True,
        help="Dynasearch (kombinierte Züge) gegen Sequentiell (ein Zug je Iteration) - bei 60 Stopps und 200 Tausend Vorschlägen liegt Dynasearch im Mittel bei 3.75 %, sequentiell bei 6.33 % über der Schranke.",
    )
    budget = st.select_slider(
        "Budget (bewertete Nachbarn)", options=list(C.BUDGETS), key="budget_select", format_func=_fmt_int,
        help="Dynasearch gewinnt bei JEDEM gemessenen Budget klar (10 Tausend: 5.81 % gegen 7.85 %; 2 Millionen: 1.25 % gegen 1.91 %) - die durchgehendste Budget-Geschichte der ganzen Linie, kein Umschlagpunkt wie bei GRASP oder Lin-Kernighan.",
    )
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed für die Lage der Stopps.")
    chain_seed = st.number_input(
        "Zufalls-Seed der Kette", *bounds("chain_seed_input"), key="chain_seed_input", step=1,
        help="Steuert die zufälligen Startlösungen der Neustarts - beide Verfahren sind sonst deterministisch (immer die gewinn-maximale Kombination bzw. der beste Zug, kein Zufall im Kern).",
    )
    st.button("🎲 Neue Kette würfeln", width="stretch", on_click=randomize_chain_seed, help="Würfelt einen neuen Seed für dieselbe Instanz.")

sync_query_params({
    "n_slider": int(n_stops), "ballung_slider": int(cluster_share), "seed_input": int(seed), "mode_radio": mode,
    "budget_select": int(budget), "chain_seed_input": int(chain_seed),
})

settings = Settings(int(n_stops), int(cluster_share), int(seed), mode, int(budget), int(chain_seed))
with st.spinner("Rechne..."):
    a = _analysis(settings)
    inst, D, bound, run = _run_with_steps(settings)
xy = inst.xy
code = verdict(a)
other_label = C.MODE_LABELS[EV.OTHER_MODE[settings.mode]]
data_key = settings

# --- Dynasearch in Aktion -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Dynasearch in Aktion")
STEP_LABELS = {1: "1 · Instanz", 2: "2 · Abstieg", 3: "3 · Ergebnis"}
if "dyna_step" not in st.session_state or st.session_state.get("dyna_step_owner") != data_key:
    st.session_state["dyna_step"] = 1
    st.session_state["dyna_step_owner"] = data_key
    st.session_state.pop("dyna_iter", None)
step_col, play_col = st.columns([5, 2])
with step_col:
    step = st.select_slider("Schritt", options=list(STEP_LABELS), key="dyna_step", format_func=lambda s: STEP_LABELS[s])
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

n_snaps = len(run.steps)
iteration = n_snaps - 1
play_search = False
if step == 2 and n_snaps > 1:
    it_col, itplay_col = st.columns([5, 2])
    with it_col:
        iteration = st.slider("Iteration", 0, n_snaps - 1, value=n_snaps - 1, key="dyna_iter", help="Die Tour nach dieser Iteration (0 = Startlösung, vor der ersten Iteration).")
    with itplay_col:
        play_search = st.button("▶️ Abstieg abspielen", width="stretch")
view_slot = st.empty()


def _frames():
    if n_snaps <= 1:
        return [0]
    return sorted({int(round(x)) for x in np.linspace(0, n_snaps - 1, min(n_snaps, 40))})


def _render(current_step, it):
    with view_slot.container():
        if current_step == 1:
            st.markdown(f"**{inst.n} Kundenstopps und das Depot (Stern)** – {inst.cluster_share} % der Stopps in Gruppen")
            st.plotly_chart(build_instance(xy), width="stretch", key="s1_map")
        elif current_step == 2:
            trace_iter = [0] + [i + 1 for i in range(len(run.steps) - 1)]
            trace_len = [s.length for s in run.steps]
            st.markdown(f"**Länge der Tour über die Dynasearch-Iterationen** ({C.MODE_LABELS[settings.mode]})")
            st.plotly_chart(build_trace(trace_iter, trace_len, bound, C.MODE_LABELS[settings.mode], "#54a24b" if settings.mode == "dynasearch" else "#f58518"), width="stretch", key=f"s2_trace_{it}")
            changed = _changed_nodes(run.steps[it - 1].tour, run.steps[it].tour) if it > 0 else None
            moves_here = run.steps[it].move[1] if (it > 0 and settings.mode == "dynasearch" and run.steps[it].move) else (1 if it > 0 else 0)
            st.markdown(f"**Tour nach Iteration {it} von {n_snaps - 1}**" + (f" - {moves_here} Zug/Züge gleichzeitig" if it > 0 else ""))
            st.plotly_chart(build_tour(xy, run.steps[it].tour, ghost=run.steps[it - 1].tour if it > 0 else None, changed_nodes=changed), width="stretch", key=f"s2_map_{it}")
            if settings.mode == "dynasearch" and n_snaps > 1:
                moves_per_iter = [s.move[1] if s.move else 0 for s in run.steps[1:]]
                st.markdown("**Gleichzeitig kombinierte Züge je Iteration**")
                st.plotly_chart(build_moves_per_iteration(moves_per_iter), width="stretch", key=f"s2_moves_{it}")
        else:
            c1, c2 = st.columns(2)
            c1.markdown(f"**{C.MODE_LABELS[settings.mode]}: beste Tour** – {a.gap:.1f} % über der Schranke")
            c1.plotly_chart(build_tour(xy, a.run_tour), width="stretch", key="s3_run")
            c2.markdown(f"**{other_label}** ({a.other_starts} Abstiege, gleiches Budget) – {a.other_gap:.1f} % über der Schranke")
            c2.plotly_chart(build_tour(xy, a.other_tour), width="stretch", key="s3_other")


if auto_play:
    for s in STEP_LABELS:
        if s == 2:
            for f in _frames():
                _render(2, f)
                time.sleep(0.15)
            time.sleep(0.6)
        else:
            _render(s, iteration)
            time.sleep(1.2)
    step = 3
elif play_search:
    for f in _frames():
        _render(2, f)
        time.sleep(0.15)
else:
    _render(step, iteration)

if step == 1:
    st.caption(f"{inst.n} Stopps; die untere Schranke der kürzesten Rundtour liegt bei {bound:,.0f} km (1-Baum-Schranke, Held-Karp).".replace(",", "."))
elif step == 2:
    moves_total = run.n_moves if settings.mode == "dynasearch" else (n_snaps - 1)
    st.caption(f"{_fmt_int(run.evaluations)} bewertete Nachbarn in {n_snaps - 1} Iterationen, {moves_total} einzelne Züge insgesamt kombiniert (Ø {moves_total / max(n_snaps - 1, 1):.1f} je Iteration).")
else:
    st.caption(f"Links die beste Tour aus {a.run_starts} Abstiegen ({C.MODE_LABELS[settings.mode]}), rechts die beste Tour aus {a.other_starts} Abstiegen ({other_label}) mit demselben Bewertungsbudget ({_fmt_int(settings.budget)}).")

st.markdown("---")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Was die Suche gefunden hat")
st.caption(
    "**Abstand zur Schranke:** Länge der Tour gegenüber einer unteren Schranke der kürzesten Rundtour (1-Baum, Held-Karp) in Prozent. "
    "Ein Lauf ist eine Ziehung (nur die Startlösungen der Neustarts streuen, beide Verfahren sind sonst deterministisch): Vergleiche gelten für diesen Lauf."
)
m1, m2, m3 = st.columns(3)
m1.metric(f"{C.MODE_LABELS[settings.mode]}: beste Tour", f"{a.gap:.1f} %", delta=f"{a.run_starts} Abstiege", delta_color="off", help="Abstand zur Schranke der besten je gefundenen Tour über alle Neustarts.")
m2.metric(other_label, f"{a.other_gap:.1f} %", delta=f"{a.other_starts} Abstiege, gleiches Budget", delta_color="off", help="Derselbe Vergleich mit dem jeweils anderen Modus bei gleichem Budget.")
m3.metric("Ein einzelner Abstieg", f"{a.single_gap:.1f} %", delta=f"{_fmt_int(a.single_evaluations)} Bewertungen", delta_color="off", help="Ein Abstieg im gewählten Modus (kein Neustart) bis zur Konvergenz.")

if code == "beats_hc":
    st.success(f"✅ {C.MODE_LABELS[settings.mode]} gewinnt: {a.gap:.1f} % über der Schranke gegen {a.other_gap:.1f} % für {other_label} (gleiches Budget). Andere Ketten streuen um dieses Ergebnis.")
elif code == "comparable":
    st.info(f"ℹ️ Gleichauf: {C.MODE_LABELS[settings.mode]} {a.gap:.1f} %, {other_label} {a.other_gap:.1f} % über der Schranke. Eine andere Kette kann das Bild drehen.")
else:
    st.warning(f"⚠️ {other_label} ist hier besser: {a.other_gap:.1f} % gegen {a.gap:.1f} % über der Schranke bei gleichem Budget. Bei großen Instanzen kann sich das Bild umdrehen - siehe README 'Was nicht funktioniert hat'.")

d1, d2 = st.columns(2)
with d1:
    st.markdown("**Kennzahlen im Detail**")
    unit_time = lambda sec: f"{sec * 1000:.0f} ms"  # noqa: E731
    st.table({"": ["Länge (km)", "Abstand zur Schranke", "Bewertete Nachbarn", "Rechenzeit"],
              f"{C.MODE_LABELS[settings.mode]}": [f"{T.tour_length(a.run_tour, D):.1f}", f"{a.gap:.2f} %", _fmt_int(settings.budget), unit_time(a.seconds)],
              other_label: [f"{T.tour_length(a.other_tour, D):.1f}", f"{a.other_gap:.2f} %", _fmt_int(settings.budget), unit_time(a.other_seconds)]})
with d2:
    st.markdown("**Was gerechnet wurde**")
    st.table({"": ["Modus", "Budget", "Neustarts", "Kreuzungen der besten Tour"],
              "Einstellung": [C.MODE_LABELS[settings.mode], _fmt_int(settings.budget), f"{a.run_starts}", f"{a.crossings_end}"]})
    st.caption("Ein Vorschlag ist ein geprüftes (i,j)-Paar aus der vollen Delta-Matrix - dieselbe Einheit für beide Modi. Rechenzeiten hängen vom Rechner ab, nur die Größenordnung zählt.")

st.markdown("---")

# --- Sweeps ------------------------------------------------------------------------------------------------------------------------------------

st.subheader("📐 Wie stark hängt das Ergebnis von Budget und Instanz ab?")
sweep_param = st.selectbox("Welcher Regler soll durchgefahren werden?", list(SWEEP_LABELS), format_func=lambda k: SWEEP_LABELS[k], key="sweep_select")
base_sweep = replace(settings, seed=0, chain_seed=0)
if st.button("Sweep über 5 feste Instanzen berechnen (dauert etwa 10 bis 60 Sekunden)", key="sweep_start"):
    st.session_state["sweep_done"] = st.session_state.get("sweep_done", set()) | {(sweep_param, base_sweep)}
if (sweep_param, base_sweep) in st.session_state.get("sweep_done", set()):
    with st.spinner("Rechne den Sweep über 5 feste Instanzen × 3 Ketten..."):
        rows_sweep = _sweep(sweep_param, base_sweep)
    st.plotly_chart(build_sweep(rows_sweep, SWEEP_LABELS[sweep_param]), width="stretch", key="sweep_chart")
    st.caption("Mittel und Streuung (Band) über 5 feste Instanzen (Seeds 100000–100004, getrennt vom Seed oben) mit je drei Ketten; alle anderen Regler wie in der Seitenleiste. "
               "Gepunktet: der jeweils andere Modus, gleiches Budget.")

st.markdown("---")

# --- Experimente --------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Ein Abstieg: wie effizient ist Dynasearch wirklich?")
if st.button("Einen einzelnen Abstieg je Modus über die Instanzgröße vergleichen (dauert etwa 15 Sekunden)", key="single_start"):
    st.session_state["single_on"] = True
if st.session_state.get("single_on"):
    with st.spinner("Rechne je einen Abstieg (kein Neustart) über 5 Instanzen × 3 Ketten, mehrere Größen..."):
        rows_single = []
        for n_val in (20, 60, 100, 150, 200):
            r = _single_descent(replace(base_sweep, n=n_val))
            rows_single.append({"n": n_val, **r})
    st.table({"Stopps": [f"{r['n']}" for r in rows_single],
              "Dynasearch: Bewertungen": [_fmt_int(r["dynasearch"]["evaluations"]) for r in rows_single],
              "Sequentiell: Bewertungen": [_fmt_int(r["sequential"]["evaluations"]) for r in rows_single],
              "Dynasearch: Güte (%)": [f"{r['dynasearch']['gap']:.2f}" for r in rows_single],
              "Sequentiell: Güte (%)": [f"{r['sequential']['gap']:.2f}" for r in rows_single]})
    st.plotly_chart(
        build_moves_per_iteration(
            [r["dynasearch"]["moves"] / max(r["dynasearch"]["iterations"], 1) for r in rows_single],
            x=[r["n"] for r in rows_single], x_title="Stopps",
        ),
        width="stretch", key="moves_chart",
    )
    st.caption("Ein einzelner Abstieg (kein Neustart) bis zur Konvergenz, Mittel über 5 feste Instanzen × 3 Ketten. Dynasearch braucht IMMER deutlich weniger Bewertungen (der Faktor wächst mit der Instanzgröße, bei 200 Stopps rund 3.9x weniger) - "
               "aber die erreichte GÜTE dreht sich: bei kleinen/mittleren Instanzen findet Dynasearch das bessere Lokaloptimum, ab etwa 80 Stopps ist sequentiell leicht genauer (ehrlicher Befund, siehe README). Das Balkendiagramm zeigt die mittlere Zahl "
               "gleichzeitig kombinierter Züge je Dynasearch-Iteration - sie wächst mit der Instanzgröße.")

st.markdown("---")

st.subheader("🔬 Budget: bleibt der Vorsprung bestehen?")
if st.button("Budget von 10 Tausend bis 2 Millionen durchfahren (dauert etwa 30 Sekunden)", key="budget_start"):
    st.session_state["budget_on"] = True
if st.session_state.get("budget_on"):
    with st.spinner("Rechne 8 Budgets × 5 Instanzen × 3 Ketten..."):
        rows_b = _sweep("budget", base_sweep)
    st.plotly_chart(build_budget(rows_b), width="stretch", key="budget_chart")
    st.table({"Budget": [_fmt_int(r["value"]) for r in rows_b], "Dynasearch (%)": [f"{r['gap']:.2f}" for r in rows_b], "Sequentiell (%)": [f"{r['other']:.2f}" for r in rows_b]})
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (60 Stopps, Neustarts). Dynasearch gewinnt bei JEDEM gemessenen Budget klar (10 Tausend: **5.81 %** gegen 7.85 %; 2 Millionen: **1.25 %** gegen 1.91 %) - "
               "die durchgehendste Budget-Geschichte der ganzen Linie, kein Umschlagpunkt wie bei GRASP oder Lin-Kernighan (bei 60 Stopps - bei größeren Instanzen sieht das anders aus, siehe Skalierung).")

st.markdown("---")

st.subheader("🔬 Streuung: wie verlässlich ist eine Kette?")
if st.button("20 Ketten auf dieser Instanz berechnen (dauert etwa 10 Sekunden)", key="spread_start"):
    st.session_state["spread_on"] = True
if st.session_state.get("spread_on"):
    with st.spinner("Rechne 20 Ketten je Modus..."):
        sp = _spread(replace(settings, chain_seed=0))
    st.plotly_chart(build_spread(sp["run"], sp["other"], C.MODE_LABELS[settings.mode], other_label), width="stretch", key="spread_chart")
    s1, s2 = st.columns(2)
    s1.metric(f"{C.MODE_LABELS[settings.mode]}: Mittel ± Streuung", f"{sp['run'].mean():.2f} ± {sp['run'].std():.2f} %", help="Mittel und Standardabweichung des Abstands der besten Tour über 20 Ketten.")
    s2.metric(f"{other_label}: Mittel ± Streuung", f"{sp['other'].mean():.2f} ± {sp['other'].std():.2f} %", help="Dieselben 20 Ketten, der jeweils andere Modus.")
    st.caption("Dieselbe Instanz, 20 verschiedene Ketten-Seeds (steuern nur die Startlösungen der Neustarts).")

st.markdown("---")

st.subheader("🔬 Skalierung: wie viel Budget braucht ein größeres Problem?")
if st.button("Stopps von 20 bis 200 durchfahren (dauert etwa 60 Sekunden)", key="scaling_start"):
    st.session_state["scaling_on"] = True
if st.session_state.get("scaling_on"):
    with st.spinner("Rechne 6 Größen × 2 Budgetregeln × 5 Instanzen × 3 Ketten..."):
        sc = _scaling(replace(base_sweep, n=C.DEFAULT_N))
    st.plotly_chart(build_scaling(sc), width="stretch", key="scaling_chart")
    st.caption("Mittel über 5 feste Instanzen × 3 Ketten (Einstellungen wie in der Seitenleiste außer Stopps und Budget). Bei festem 200-Tausend-Budget gewinnt Dynasearch bis 60 Stopps klar, dreht sich aber AB 100 Stopps um "
               "(sequentiell 8.78 % gegen 9.74 % bei 100 Stopps) - bei so wenig Budget pro Stopp passt für beide kaum ein Neustart, und dann zählt nur noch die (bei großen Instanzen leicht schlechtere) Güte eines einzelnen Dynasearch-Abstiegs.")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die Instanz ist klein bis mittelgroß** | Bei festem 200-Tausend-Budget gewinnt Dynasearch bis 60 Stopps klar, verliert aber ab 100 Stopps knapp gegen sequentiell (9.74 % gegen 8.78 % bei 100 Stopps, 10.03 % gegen 9.56 % bei 200) - ein einzelner Dynasearch-Abstieg findet bei großen Instanzen ein leicht schlechteres Lokaloptimum als ein einzelner sequentieller. | Mehr Budget (der Umschlagpunkt verschiebt sich nach hinten, kehrt sich aber auch dann irgendwann um) |
| **Genug Bewertungen für mehr als einen Neustart** | Bei sehr großen Instanzen und knappem Budget passt für beide Verfahren kaum ein Neustart - dann zählt nur die Güte EINES Abstiegs, nicht mehr die Effizienz-Ersparnis von Dynasearch. | (kein Nachfolger nötig - dieselbe Lehre wie bei jedem restart-basierten Stück dieser Linie) |
| **Kein Zug referenziert das Depot** | Die DP schließt Züge, die Position 0 betreffen, bewusst aus (wie im Original-Paper) - ohne den anschließenden sequentiellen Aufräum-Durchgang wäre das Ergebnis KEIN echtes 2-opt-Lokaloptimum (geprüft, siehe Tests). | (kein Nachfolger nötig - der Aufräum-Durchgang behebt es vollständig) |
| **Nur 2-opt-Züge, keine Or-opt-Kombination** | Die Unabhängigkeits-DP ist nur für 2-opt-Segmentumkehrungen hergeleitet; eine Erweiterung auf Or-opt-Züge bräuchte eine eigene Kompatibilitäts-Definition - bewusst nicht umgesetzt. | **VRP-Nachbarschaften** (inter-route-Züge, sobald CVRP-Infrastruktur existiert) |
"""
)
st.caption(
    "Die Nachbarn des Nachbarschafts-Zweigs: VRP-Nachbarschaften (inter-route-Züge) setzen auf Dynasearch und Lin-Kernighan auf, sobald CVRP-Infrastruktur existiert; "
    "ALNS ist der Konvergenzpunkt mit VNS."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Problem.** Kürzeste Rundtour über $N = n+1$ Knoten mit euklidischen Entfernungen $d_{ij}$; $L(\pi)$ ist die Länge einer Tour $\pi$.

**2-opt-Zug.** Ein Zug $(i,j)$ entfernt die Kanten $(t_i,t_{i+1})$ und $(t_j,t_{j+1})$, fügt $(t_i,t_j)$ und $(t_{i+1},t_{j+1})$ ein
(Stück zwischen $i+1$ und $j$ umgekehrt); referenziert die vier Positionen $i, i+1, j, j+1$.

**Unabhängigkeit.** Zwei Züge $(i_1,j_1)$ und $(i_2,j_2)$ mit $i_1 < i_2$ sind unabhängig genau dann, wenn $i_2 \ge j_1 + 2$ (ihre
Fußabdrücke $[i,j+1]$ überlappen sich nicht) - dann ist der Gesamtgewinn additiv: $\Delta(\pi, \{\text{Zug}_1,\text{Zug}_2\}) =
\Delta(\pi,\text{Zug}_1) + \Delta(\pi,\text{Zug}_2)$.

**Dynasearch-DP.** Gesucht die gewinn-maximale Menge paarweise unabhängiger, verbessernder Züge (Züge, die Position 0
referenzieren, ausgeschlossen - wie im Original-Paper). Klassisches gewichtetes Intervall-Scheduling: $dp[p] = \max(dp[p-1],
\max_{(i,j): j+2=p} dp[i] + \text{Gewinn}(i,j))$, gelöst in $O(n^2)$ (Kandidaten) statt $O(2^k)$ (Teilmengen).

**Aufräum-Durchgang.** Da Dynasearch Position 0 ausschließt, garantiert die Konvergenz allein nur ein Lokaloptimum unter dieser
Einschränkung; ein abschließender sequentieller 2-opt-Durchgang (volle Nachbarschaft inkl. Depot) macht daraus ein echtes
2-opt-Lokaloptimum.

**Kennzahl.** Abstand zur Schranke $= 100 \cdot (L - w)/w$ mit der 1-Baum-Schranke $w$. Vergleichsgröße bei gleichem Budget:
sequentielles bestes-Verbesserung-2-opt (voller Rescan, ein Zug je Iteration).

**Grenzen.** (1) Kein Or-opt in der DP (nur 2-opt-Kompatibilität hergeleitet). (2) Die Güte-Effizienz-Abwägung dreht sich mit
der Instanzgröße. (3) Ohne genug Budget für mehrere Neustarts zählt nur die Güte eines einzelnen Abstiegs.

**Literatur.** Potts, C. N., & van de Velde, S. (1995). *Dynasearch - Iterative local improvement by dynamic programming:
Part I, The traveling salesman problem.* Technical Report, University of Twente. Congram, R. K., Potts, C. N., & van de
Velde, S. L. (2002). *An Iterated Dynasearch Algorithm for the Single-Machine Total Weighted Tardiness Scheduling Problem.*
INFORMS Journal on Computing, 14(1), 52-67.

Implementiert in `dyna_algorithm.py` (die DP, der Aufräum-Durchgang), `dyna_tour.py` (Nachbarschaften, Abstieg, Schranke -
aus der Hill-Climbing-Demo), `dyna_scenario.py` (Instanzen), `dyna_evaluation.py` (Kennzahlen, Sweeps, Experimente, Urteil).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
