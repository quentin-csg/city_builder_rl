"""Tests pour vitruvius.engine.victory (runtime)."""

from collections import Counter

import pytest

from vitruvius.config import load_config
from vitruvius.engine.victory import check_defeat, compute_city_level


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def city_levels(cfg):
    return cfg.city_levels.city_levels


# ---------------------------------------------------------------------------
# compute_city_level
# ---------------------------------------------------------------------------


def test_city_level_minimum_is_1_no_pop(city_levels):
    """Même sans population, le niveau minimum retourné est 1."""
    assert compute_city_level(0, 0.5, Counter(), city_levels) == 1


def test_city_level_1_insufficient_conditions(city_levels):
    """pop=10 < 50 (level 1 min) → retourne 1 quand même (plancher absolu)."""
    assert compute_city_level(10, 0.0, Counter(), city_levels) == 1


def test_city_level_2_all_conditions_met(city_levels):
    """pop=150, sat=0.40, market présent → level 2."""
    placed = Counter({"market": 1})
    assert compute_city_level(150, 0.40, placed, city_levels) == 2


def test_city_level_2_missing_market(city_levels):
    """pop=150, sat=0.40 mais pas de market → level 1."""
    assert compute_city_level(150, 0.40, Counter(), city_levels) == 1


def test_city_level_2_insufficient_sat(city_levels):
    """pop=150 mais sat=0.39 < 0.40 → level 1."""
    placed = Counter({"market": 1})
    assert compute_city_level(150, 0.39, placed, city_levels) == 1


def test_city_level_2_insufficient_pop(city_levels):
    """pop=149 < 150 avec market + sat OK → level 1."""
    placed = Counter({"market": 1})
    assert compute_city_level(149, 0.40, placed, city_levels) == 1


def test_city_level_3_all_conditions_met(city_levels):
    """pop=500, sat=0.50, market + forum + temple → level 3."""
    placed = Counter({"market": 1, "forum": 1, "temple": 1})
    assert compute_city_level(500, 0.50, placed, city_levels) == 3


def test_city_level_3_missing_temple(city_levels):
    """pop=500, sat=0.50, market + forum seulement (pas temple) → level 2."""
    placed = Counter({"market": 1, "forum": 1})
    assert compute_city_level(500, 0.50, placed, city_levels) == 2


def test_city_level_3_missing_forum(city_levels):
    """pop=500, sat=0.50, market + temple (pas forum) → level 2."""
    placed = Counter({"market": 1, "temple": 1})
    assert compute_city_level(500, 0.50, placed, city_levels) == 2


def test_city_level_4_all_conditions_met(city_levels):
    """pop=1500, sat=0.60, héritage complet + theater + baths → level 4."""
    placed = Counter({"market": 1, "forum": 1, "temple": 1, "theater": 1, "baths": 1})
    assert compute_city_level(1500, 0.60, placed, city_levels) == 4


def test_city_level_4_missing_one_building(city_levels):
    """pop=1500, sat=0.60, market + forum + temple + theater (pas baths) → level 3."""
    placed = Counter({"market": 1, "forum": 1, "temple": 1, "theater": 1})
    assert compute_city_level(1500, 0.60, placed, city_levels) == 3


def test_city_level_4_cannot_skip_level_3(city_levels):
    """Avec theater + baths mais sans forum/temple → impossible d'atteindre L4 (héritage)."""
    placed = Counter({"market": 1, "theater": 1, "baths": 1})
    assert compute_city_level(1500, 0.60, placed, city_levels) == 2


def test_city_level_5_nova_roma(city_levels):
    """pop=2500, sat=0.65, tous bâtiments hérités + obelisk + prefecture → level 5."""
    placed = Counter({
        "market": 1, "forum": 1, "temple": 1, "theater": 1, "baths": 1,
        "obelisk": 1, "prefecture": 1,
    })
    assert compute_city_level(2500, 0.65, placed, city_levels) == 5


def test_city_level_5_insufficient_sat(city_levels):
    """pop=2500 mais sat=0.60 < 0.65 → level 4."""
    placed = Counter({
        "market": 1, "forum": 1, "temple": 1, "theater": 1, "baths": 1,
        "obelisk": 1, "prefecture": 1,
    })
    assert compute_city_level(2500, 0.60, placed, city_levels) == 4


def test_city_level_5_missing_obelisk(city_levels):
    """pop=2500, sat=0.65, tous bâtiments sauf obelisk → level 4."""
    placed = Counter({
        "market": 1, "forum": 1, "temple": 1, "theater": 1, "baths": 1,
        "prefecture": 1,
    })
    assert compute_city_level(2500, 0.65, placed, city_levels) == 4


def test_city_level_multiple_same_building(city_levels):
    """Plusieurs instances du même bâtiment comptent comme présence (Counter > 0)."""
    placed = Counter({"market": 3})  # 3 marchés
    assert compute_city_level(150, 0.40, placed, city_levels) == 2


# ---------------------------------------------------------------------------
# check_defeat
# ---------------------------------------------------------------------------


def test_check_defeat_pop_zero_with_housing():
    """Population nulle avec logement posé → défaite (famine/exode a tout tué)."""
    assert check_defeat(0, 0, has_housing=True) is True


def test_check_defeat_pop_zero_no_housing():
    """Population nulle sans logement → pas de défaite (début de partie)."""
    assert check_defeat(0, 0, has_housing=False) is False


def test_check_defeat_bankrupt_5():
    """5 tours consécutifs en banqueroute → défaite."""
    assert check_defeat(100, 5) is True


def test_check_defeat_bankrupt_6():
    """6 tours → toujours défaite."""
    assert check_defeat(100, 6) is True


def test_check_defeat_bankrupt_4():
    """4 tours consécutifs → pas encore défaite."""
    assert check_defeat(100, 4) is False


def test_check_defeat_healthy():
    """Pop > 0, banqueroute = 0 → pas de défaite."""
    assert check_defeat(500, 0) is False


def test_check_defeat_pop_zero_overrides_bankrupt():
    """Pop=0 avec logement + bankrupt=0 → défaite (la pop prime)."""
    assert check_defeat(0, 0, has_housing=True) is True
