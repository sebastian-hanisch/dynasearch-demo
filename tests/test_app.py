"""AppTest-Rauchtests: Voreinstellung, jedes Preset, jeder Schritt, Randwerte, Würfel-Knöpfe, Permalink-Grenzen, Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import dyna_constants as C
import dyna_evaluation as ev

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(dyna_step=1, **state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    if dyna_step != 1:
        at.select_slider(key="dyna_step").set_value(dyna_step).run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def _metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


def test_default_run_has_no_exception_and_shows_the_measured_default():
    at = _run()
    _ok(at)
    assert _metric(at, "Dynasearch (kombinierte Züge): beste Tour") == "2.2 %"
    assert _metric(at, "Sequentiell (ein Zug je Iteration)") == "4.2 %"
    assert _metric(at, "Ein einzelner Abstieg") == "2.2 %"
    assert any("gewinnt" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = C.PRESETS[name]
    assert at.session_state["mode_radio"] == p["mode"] and at.session_state["budget_select"] == p["budget"] and at.session_state["n_slider"] == p["n"]
    assert at.metric


@pytest.mark.parametrize("step", [1, 2, 3])
@pytest.mark.parametrize("mode", ["dynasearch", "sequential"])
def test_every_step_runs(step, mode):
    at = _run(n_slider=20, budget_select=25000, mode_radio=mode, dyna_step=step)
    _ok(at)
    assert at.get("plotly_chart") and at.session_state["dyna_step"] == step


def test_step_two_iteration_slider_and_play_descent():
    at = _run(dyna_step=2, budget_select=25000)
    _ok(at)
    lv = next(s for s in at.slider if s.key == "dyna_iter")
    assert lv.value == lv.max
    lv.set_value(0).run()
    _ok(at)
    at2 = _run(dyna_step=2, budget_select=25000)
    next(b for b in at2.button if b.label == "▶️ Abstieg abspielen").click().run()
    _ok(at2)


def test_play_runs_through_all_steps_without_duplicate_keys():
    at = _run(n_slider=20, budget_select=10000)
    next(b for b in at.button if b.label == "▶️ Abspielen").click().run()
    _ok(at)


def test_sequential_mode_has_no_iteration_slider_for_a_single_move():
    """Sequentiell wendet pro Iteration genau einen Zug an - bei sehr kleinem Budget bleibt oft nur ein Snapshot,
    dann zeigt die Schritt-2-Ansicht keinen Iterations-Regler (n_snaps <= 1)."""
    at = _run(n_slider=10, budget_select=1, mode_radio="sequential", dyna_step=2)
    _ok(at)


def test_dice_buttons_change_the_seeds():
    at = _run(budget_select=10000)
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Instanz generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old
    old_c = at.session_state["chain_seed_input"]
    next(b for b in at.button if b.label == "🎲 Neue Kette würfeln").click().run()
    _ok(at)
    assert at.session_state["chain_seed_input"] != old_c


@pytest.mark.parametrize("kw", [dict(n_slider=200, budget_select=50000), dict(n_slider=10, ballung_slider=100, budget_select=10000),
                                 dict(mode_radio="sequential", budget_select=25000), dict(n_slider=10, budget_select=10000, mode_radio="sequential")])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_permalink_values_are_clamped_and_snapped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["ballung"] = "40"
    at.query_params["mode"] = "not_a_mode"
    at.query_params["budget"] = "12345"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["ballung_slider"] == 50
    assert at.session_state["mode_radio"] == C.DEFAULT_MODE and at.session_state["budget_select"] == C.DEFAULT_BUDGET


def test_sweeps_run_on_demand():
    at = _run(n_slider=10, budget_select=10000)
    at.selectbox(key="sweep_select").set_value("n").run()
    next(b for b in at.button if b.key == "sweep_start").click().run()
    _ok(at)
    assert at.get("plotly_chart")


def test_experiments_run_on_demand(monkeypatch):
    monkeypatch.setitem(ev.SWEEP_VALUES, "budget", (2000, 5000))
    monkeypatch.setattr(C, "SCALING_N", (10, 20))
    monkeypatch.setattr(ev, "SCALING_POLICIES", (("Budget 4 Tausend", lambda n: 4000), ("Budget 300 · Stopps", lambda n: 300 * n)))
    at = _run(n_slider=10, budget_select=10000)
    for key, flag in (("single_start", "single_on"), ("budget_start", "budget_on"), ("spread_start", "spread_on"), ("scaling_start", "scaling_on")):
        next(b for b in at.button if b.key == key).click().run()
        _ok(at)
        assert at.session_state[flag]


def test_footer_and_grenzen_are_present():
    at = _run(budget_select=10000)
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    assert any("Die Instanz ist klein bis mittelgroß" in m.value for m in at.markdown)
