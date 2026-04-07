"""Tests unitaires pour vitruvius.rl.reward.compute_reward."""

import pytest

from vitruvius.engine.turn import TurnResult
from vitruvius.rl.reward import RewardState, compute_reward


# ---------------------------------------------------------------------------
# Fixture : TurnResult et RewardState neutres (aucun effet)
# ---------------------------------------------------------------------------

def neutral_result(**overrides) -> TurnResult:
    """TurnResult sans aucun flag actif. Overrides appliqués par dessus."""
    base = dict(
        production={},
        taxes_collected=0.0,
        maintenance_paid=0.0,
        passive_income=0.0,
        famine_count=0,
        famine_pop_lost=0,
        evolved=0,
        regressed=0,
        growth=0,
        exodus=0,
        new_event=None,
        global_satisfaction=0.5,
        total_population=0,
        city_level=1,
        done=False,
        victory=False,
        defeat=False,
        bankrupt=False,
    )
    base.update(overrides)
    return TurnResult(**base)


def same_state(pop: int = 0, level: int = 1, sat: float = 0.5, housing: int = 0) -> RewardState:
    """Crée un RewardState avec les valeurs données (prev et curr identiques → deltas = 0)."""
    return RewardState(
        total_population=pop,
        city_level=level,
        global_satisfaction=sat,
        housing_sum=housing,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reward_survival_only():
    """Aucun delta, aucun flag → reward == 0.01 (survie seule)."""
    prev = same_state()
    curr = same_state()
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(0.01, abs=1e-6)


def test_reward_delta_pop_positive():
    """+100 pop → contribution +1.0 (+ 0.01 survie)."""
    prev = same_state(pop=0)
    curr = same_state(pop=100)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(1.0 + 0.01, abs=1e-6)


def test_reward_delta_pop_negative():
    """-50 pop → contribution -0.5."""
    prev = same_state(pop=100)
    curr = same_state(pop=50)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(-0.5 + 0.01, abs=1e-6)


def test_reward_delta_city_level():
    """+1 city level → contribution +5.0."""
    prev = same_state(level=1)
    curr = same_state(level=2)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(5.0 + 0.01, abs=1e-6)


def test_reward_delta_city_level_negative():
    """-1 city level (régression) → contribution -5.0."""
    prev = same_state(level=3)
    curr = same_state(level=2)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(-5.0 + 0.01, abs=1e-6)


def test_reward_delta_satisfaction():
    """+0.2 satisfaction → contribution +0.1."""
    prev = same_state(sat=0.5)
    curr = same_state(sat=0.7)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(0.5 * 0.2 + 0.01, abs=1e-6)


def test_reward_delta_housing_sum():
    """+20 housing_sum → contribution +0.2."""
    prev = same_state(housing=0)
    curr = same_state(housing=20)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(0.1 * (20 / 10) + 0.01, abs=1e-6)


def test_reward_bankrupt_penalty():
    """`bankrupt=True` → pénalité -0.5."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(bankrupt=True)
    assert compute_reward(prev, curr, result) == pytest.approx(-0.5 + 0.01, abs=1e-6)


def test_reward_famine_penalty():
    """`famine_count > 0` → pénalité -0.3."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(famine_count=3)
    assert compute_reward(prev, curr, result) == pytest.approx(-0.3 + 0.01, abs=1e-6)


def test_reward_exodus_penalty():
    """`exodus > 0` → pénalité -0.2."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(exodus=50)
    assert compute_reward(prev, curr, result) == pytest.approx(-0.2 + 0.01, abs=1e-6)


def test_reward_victory_bonus():
    """`victory=True` → +100 (+ survie)."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(victory=True, done=True)
    assert compute_reward(prev, curr, result) == pytest.approx(100.0 + 0.01, abs=1e-6)


def test_reward_defeat_penalty():
    """`defeat=True` → -10 (+ survie)."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(defeat=True, done=True)
    assert compute_reward(prev, curr, result) == pytest.approx(-10.0 + 0.01, abs=1e-6)


def test_reward_combined():
    """Plusieurs effets simultanés : somme exacte vérifiée."""
    # +100 pop, +1 level, famine, bankrupt
    prev = RewardState(total_population=0, city_level=1, global_satisfaction=0.5, housing_sum=0)
    curr = RewardState(total_population=100, city_level=2, global_satisfaction=0.5, housing_sum=0)
    result = neutral_result(famine_count=1, bankrupt=True)

    expected = (
        1.0 * (100 / 100)   # delta_pop
        + 5.0 * 1           # delta_level
        + 0.5 * 0.0         # delta_sat (inchangée)
        + 0.1 * 0.0         # delta_housing (inchangé)
        + (-0.5)            # bankrupt
        + (-0.3)            # famine
        + 0.01              # survie
    )
    assert compute_reward(prev, curr, result) == pytest.approx(expected, abs=1e-6)
