"""Tests unitaires pour vitruvius.rl.reward.compute_reward."""

import pytest

from vitruvius.engine.turn import TurnResult
from vitruvius.rl.reward import (
    RewardState,
    W_BANKRUPT,
    W_BUILD_FORUM,
    W_BUILD_OBELISK,
    W_BUILD_PREFECTURE,
    W_DEFEAT,
    W_FAMINE,
    W_EXODUS,
    W_FIRST_BATHS,
    W_FIRST_FARM,
    W_FIRST_GRANARY,
    W_FIRST_HOUSE,
    W_FIRST_LUMBER_CAMP,
    W_FIRST_MARBLE_QUARRY,
    W_FIRST_MARKET,
    W_FIRST_POPULATION,
    W_FIRST_TEMPLE,
    W_FIRST_THEATER,
    W_FIRST_TRADING_POST,
    W_FIRST_WAREHOUSE_MARBLE,
    W_FIRST_WELL,
    W_LEVEL,
    W_MARBLE_PROGRESS,
    W_POSITIVE_INCOME,
    W_POP,
    W_SAT,
    W_HOUSING,
    W_SURVIVAL,
    W_VICTORY,
    compute_reward,
)


# ---------------------------------------------------------------------------
# Fixtures
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


def same_state(
    pop: int = 0,
    level: int = 1,
    sat: float = 0.5,
    housing: int = 0,
    **flags,
) -> RewardState:
    """Crée un RewardState (prev et curr identiques → deltas = 0)."""
    return RewardState(
        total_population=pop,
        city_level=level,
        global_satisfaction=sat,
        housing_sum=housing,
        **flags,
    )


# ---------------------------------------------------------------------------
# Tests deltas continus
# ---------------------------------------------------------------------------

def test_reward_survival_only():
    """Aucun delta, aucun flag → reward == W_SURVIVAL == 0.0."""
    prev = same_state()
    curr = same_state()
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_SURVIVAL, abs=1e-6)


def test_reward_delta_pop_positive():
    """+100 pop → contribution W_POP * 1.0."""
    prev = same_state(pop=0)
    curr = same_state(pop=100)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_POP * 1.0 + W_SURVIVAL, abs=1e-6)


def test_reward_delta_pop_negative():
    """-50 pop → contribution W_POP * -0.5."""
    prev = same_state(pop=100)
    curr = same_state(pop=50)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_POP * -0.5 + W_SURVIVAL, abs=1e-6)


def test_reward_delta_city_level():
    """+1 city level → contribution W_LEVEL."""
    prev = same_state(level=1)
    curr = same_state(level=2)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_LEVEL + W_SURVIVAL, abs=1e-6)


def test_reward_delta_city_level_negative():
    """-1 city level → contribution -W_LEVEL."""
    prev = same_state(level=3)
    curr = same_state(level=2)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(-W_LEVEL + W_SURVIVAL, abs=1e-6)


def test_reward_delta_satisfaction():
    """+0.2 satisfaction → contribution W_SAT * 0.2."""
    prev = same_state(sat=0.5)
    curr = same_state(sat=0.7)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_SAT * 0.2 + W_SURVIVAL, abs=1e-6)


def test_reward_delta_housing_sum():
    """+20 housing_sum → contribution W_HOUSING * 2.0."""
    prev = same_state(housing=0)
    curr = same_state(housing=20)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_HOUSING * 2.0 + W_SURVIVAL, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests pénalités
# ---------------------------------------------------------------------------

def test_reward_bankrupt_penalty():
    """`bankrupt=True` → pénalité W_BANKRUPT."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(bankrupt=True)
    assert compute_reward(prev, curr, result) == pytest.approx(W_BANKRUPT + W_SURVIVAL, abs=1e-6)


def test_reward_famine_penalty():
    """`famine_count > 0` → pénalité W_FAMINE."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(famine_count=3)
    assert compute_reward(prev, curr, result) == pytest.approx(W_FAMINE + W_SURVIVAL, abs=1e-6)


def test_reward_exodus_penalty():
    """`exodus > 0` → pénalité W_EXODUS."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(exodus=50)
    assert compute_reward(prev, curr, result) == pytest.approx(W_EXODUS + W_SURVIVAL, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests terminaison
# ---------------------------------------------------------------------------

def test_reward_victory_bonus():
    """`victory=True` → +W_VICTORY."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(victory=True, done=True)
    assert compute_reward(prev, curr, result) == pytest.approx(W_VICTORY + W_SURVIVAL, abs=1e-6)


def test_reward_defeat_penalty():
    """`defeat=True` → W_DEFEAT."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(defeat=True, done=True)
    assert compute_reward(prev, curr, result) == pytest.approx(W_DEFEAT + W_SURVIVAL, abs=1e-6)


# ---------------------------------------------------------------------------
# Tests milestones one-shot
# ---------------------------------------------------------------------------

def test_reward_first_house_milestone():
    """Transition first_house_placed False→True → +W_FIRST_HOUSE."""
    prev = same_state(first_house_placed=False)
    curr = same_state(first_house_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_HOUSE + W_SURVIVAL, abs=1e-6)


def test_reward_first_farm_milestone():
    """Transition first_farm_placed False→True → +W_FIRST_FARM."""
    prev = same_state(first_farm_placed=False)
    curr = same_state(first_farm_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_FARM + W_SURVIVAL, abs=1e-6)


def test_reward_first_well_milestone():
    """Transition first_well_placed False→True → +W_FIRST_WELL."""
    prev = same_state(first_well_placed=False)
    curr = same_state(first_well_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_WELL + W_SURVIVAL, abs=1e-6)


def test_reward_first_temple_milestone():
    """Transition first_temple_placed False→True → +W_FIRST_TEMPLE."""
    prev = same_state(first_temple_placed=False)
    curr = same_state(first_temple_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_TEMPLE + W_SURVIVAL, abs=1e-6)


def test_reward_build_forum():
    """Transition has_forum False→True → +W_BUILD_FORUM."""
    prev = same_state(has_forum=False)
    curr = same_state(has_forum=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_BUILD_FORUM + W_SURVIVAL, abs=1e-6)


def test_reward_build_prefecture():
    """Transition has_prefecture False→True → +W_BUILD_PREFECTURE."""
    prev = same_state(has_prefecture=False)
    curr = same_state(has_prefecture=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_BUILD_PREFECTURE + W_SURVIVAL, abs=1e-6)


def test_reward_build_obelisk():
    """Transition has_obelisk False→True → +W_BUILD_OBELISK."""
    prev = same_state(has_obelisk=False)
    curr = same_state(has_obelisk=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_BUILD_OBELISK + W_SURVIVAL, abs=1e-6)


def test_reward_first_granary_milestone():
    """Transition first_granary_placed False→True → +W_FIRST_GRANARY."""
    prev = same_state(first_granary_placed=False)
    curr = same_state(first_granary_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_GRANARY + W_SURVIVAL, abs=1e-6)


def test_reward_first_market_milestone():
    """Transition first_market_placed False→True → +W_FIRST_MARKET."""
    prev = same_state(first_market_placed=False)
    curr = same_state(first_market_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_MARKET + W_SURVIVAL, abs=1e-6)


def test_reward_first_lumber_camp_milestone():
    """Transition first_lumber_camp_placed False→True → +W_FIRST_LUMBER_CAMP."""
    prev = same_state(first_lumber_camp_placed=False)
    curr = same_state(first_lumber_camp_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_LUMBER_CAMP + W_SURVIVAL, abs=1e-6)


def test_reward_first_trading_post_milestone():
    """Transition first_trading_post_placed False→True → +W_FIRST_TRADING_POST."""
    prev = same_state(first_trading_post_placed=False)
    curr = same_state(first_trading_post_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_TRADING_POST + W_SURVIVAL, abs=1e-6)


def test_reward_first_population_milestone():
    """Transition first_population False→True → +W_FIRST_POPULATION."""
    prev = same_state(first_population=False)
    curr = same_state(first_population=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_POPULATION + W_SURVIVAL, abs=1e-6)


def test_reward_positive_income_bonus():
    """`taxes + passive > maintenance` → +W_POSITIVE_INCOME."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(taxes_collected=30.0, passive_income=20.0, maintenance_paid=10.0)
    assert compute_reward(prev, curr, result) == pytest.approx(W_POSITIVE_INCOME + W_SURVIVAL, abs=1e-6)


def test_reward_no_positive_income_when_loss():
    """`taxes + passive <= maintenance` → pas de bonus W_POSITIVE_INCOME."""
    prev = same_state()
    curr = same_state()
    result = neutral_result(taxes_collected=5.0, passive_income=5.0, maintenance_paid=20.0)
    assert compute_reward(prev, curr, result) == pytest.approx(W_SURVIVAL, abs=1e-6)


def test_reward_first_marble_quarry_milestone():
    """Transition first_marble_quarry_placed False→True → +W_FIRST_MARBLE_QUARRY."""
    prev = same_state(first_marble_quarry_placed=False)
    curr = same_state(first_marble_quarry_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_MARBLE_QUARRY + W_SURVIVAL, abs=1e-6)


def test_reward_first_warehouse_marble_milestone():
    """Transition first_warehouse_marble_placed False→True → +W_FIRST_WAREHOUSE_MARBLE."""
    prev = same_state(first_warehouse_marble_placed=False)
    curr = same_state(first_warehouse_marble_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_WAREHOUSE_MARBLE + W_SURVIVAL, abs=1e-6)


def test_reward_first_baths_milestone():
    """Transition first_baths_placed False→True → +W_FIRST_BATHS."""
    prev = same_state(first_baths_placed=False)
    curr = same_state(first_baths_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_BATHS + W_SURVIVAL, abs=1e-6)


def test_reward_first_theater_milestone():
    """Transition first_theater_placed False→True → +W_FIRST_THEATER."""
    prev = same_state(first_theater_placed=False)
    curr = same_state(first_theater_placed=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_FIRST_THEATER + W_SURVIVAL, abs=1e-6)


def test_reward_marble_progress_on_gain():
    """Gain de 100 marble → W_MARBLE_PROGRESS * 1.0. Ne compte pas si marble diminue."""
    prev = same_state(marble_stock=0)
    curr = same_state(marble_stock=100)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_MARBLE_PROGRESS * 1.0 + W_SURVIVAL, abs=1e-6)


def test_reward_marble_progress_zero_on_spend():
    """Dépense de marble (−100) → aucune pénalité (max 0). Le milestone du bâtiment compense."""
    prev = same_state(marble_stock=100)
    curr = same_state(marble_stock=0)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_SURVIVAL, abs=1e-6)


def test_reward_milestone_not_triggered_if_already_true():
    """Si le flag était déjà True, aucun bonus supplémentaire."""
    prev = same_state(has_forum=True)
    curr = same_state(has_forum=True)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_SURVIVAL, abs=1e-6)


def test_reward_milestone_not_triggered_false_to_false():
    """Si le flag reste False, aucun bonus."""
    prev = same_state(has_obelisk=False)
    curr = same_state(has_obelisk=False)
    result = neutral_result()
    assert compute_reward(prev, curr, result) == pytest.approx(W_SURVIVAL, abs=1e-6)


# ---------------------------------------------------------------------------
# Test combiné
# ---------------------------------------------------------------------------

def test_reward_combined():
    """Plusieurs effets simultanés : somme exacte vérifiée."""
    prev = RewardState(total_population=0, city_level=1, global_satisfaction=0.5, housing_sum=0)
    curr = RewardState(total_population=100, city_level=2, global_satisfaction=0.5, housing_sum=0,
                       first_house_placed=True)
    result = neutral_result(famine_count=1, bankrupt=True)

    expected = (
        W_POP * (100 / 100.0)   # delta_pop
        + W_LEVEL * 1            # delta_level
        + W_SAT * 0.0            # delta_sat
        + W_HOUSING * 0.0        # delta_housing
        + W_FIRST_HOUSE          # milestone first_house
        + W_BANKRUPT             # bankrupt
        + W_FAMINE               # famine
        + W_SURVIVAL
    )
    assert compute_reward(prev, curr, result) == pytest.approx(expected, abs=1e-6)
