"""Presets: Vollständigkeit, gültige Werte, Urteile über mehrere Instanzen und Ketten (Bänder), Permalink-Konstanten."""

import pytest

import dyna_constants as C
import dyna_evaluation as ev
import dyna_presets as P


def _settings(p, seed=None, chain_seed=None):
    return ev.Settings(n=p["n"], cluster_share=p["ballung"], seed=p["seed"] if seed is None else seed, mode=p["mode"],
                        budget=p["budget"], chain_seed=p["chain_seed"] if chain_seed is None else chain_seed)


def test_every_preset_has_help_bands_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) == set(C.PRESET_EXPECTED_BANDS) and len(C.PRESETS) == 5
    for name, p in C.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and C.PRESET_HELP[name]


def test_preset_values_are_valid_and_match_the_setting_specs():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and (p["n"] - C.N_MIN) % C.N_STEP == 0
        assert C.BALLUNG_MIN <= p["ballung"] <= C.BALLUNG_MAX and p["ballung"] % C.BALLUNG_STEP == 0
        assert p["mode"] in C.MODES and p["budget"] in C.BUDGETS
        for key, state_key in P.PRESET_KEYS.items():
            P.SETTING_SPECS[state_key].caster(p[key])


def test_default_preset_equals_the_default_settings():
    assert _settings(C.PRESETS["Standardfall (Voreinstellung)"]) == ev.Settings()


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_preset_verdicts_stay_in_their_bands_over_instances_and_chains(name):
    p = C.PRESETS[name]
    seeds = range(2) if p["n"] >= 150 else range(5)
    seen = {ev.verdict(ev.analyse(_settings(p, seed=seed, chain_seed=ch))) for seed in seeds for ch in (0, 1)}
    assert seen <= C.PRESET_EXPECTED_BANDS[name], seen
    assert ev.verdict(ev.analyse(_settings(p))) in C.PRESET_EXPECTED_BANDS[name]


def test_preset_contrasts_at_the_default_instance():
    """Bei der Standard-Instanz (Seed 35, Kette 0) gewinnt Dynasearch klar gegen Sequentiell (gleiches Budget); das
    'Sequentiell'-Preset ist derselbe Lauf aus der anderen Perspektive (Rollen von gap/other_gap vertauscht)."""
    g = {name: ev.analyse(_settings(p)) for name, p in C.PRESETS.items() if p["n"] == C.DEFAULT_N and p["budget"] == C.DEFAULT_BUDGET}
    std, seq = g["Standardfall (Voreinstellung)"], g["Sequentiell statt Dynasearch"]
    assert std.gap < std.other_gap                                    # Dynasearch schlägt Sequentiell bei Standard-Budget
    assert seq.gap == pytest.approx(std.other_gap) and seq.other_gap == pytest.approx(std.gap)


def test_bounds_and_snapping_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("seed_input") == (0, C.SEED_MAX)
    assert P.STEPS == {"n_slider": C.N_STEP, "ballung_slider": C.BALLUNG_STEP}
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)
    assert C.DEFAULT_BUDGET in C.BUDGETS
