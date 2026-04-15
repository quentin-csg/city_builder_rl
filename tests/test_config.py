"""Tests de chargement et validation de la configuration YAML."""

import math

import pytest
from pydantic import ValidationError

from vitruvius.config import GameConfig, load_config

EXPECTED_BUILDING_IDS = {
    "road", "well", "fountain", "aqueduct", "housing",
    "wheat_farm", "lumber_camp", "marble_quarry", "granary", "market",
    "small_altar", "temple", "baths", "theater",
    "forum", "prefecture",
    "trading_post",
    "warehouse_wood", "warehouse_marble",
    "obelisk",
}

EXPECTED_RESOURCE_IDS = {"denarii", "wheat", "wood", "marble"}

VALID_NEEDS = {"water", "food", "religion", "hygiene", "entertainment", "security"}


@pytest.fixture(scope="module")
def cfg() -> GameConfig:
    """Charge la config une seule fois pour tout le module."""
    return load_config()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_load_all_configs(cfg: GameConfig) -> None:
    assert isinstance(cfg, GameConfig)


# ---------------------------------------------------------------------------
# Ressources
# ---------------------------------------------------------------------------

def test_resources_count(cfg: GameConfig) -> None:
    assert set(cfg.resources.resources.keys()) == EXPECTED_RESOURCE_IDS


def test_resources_starting_values(cfg: GameConfig) -> None:
    r = cfg.resources.resources
    assert r["denarii"].starting_amount == 1000
    assert r["wheat"].starting_amount == 300
    assert r["wood"].starting_amount == 150
    assert r["marble"].starting_amount == 0


def test_resources_storage(cfg: GameConfig) -> None:
    r = cfg.resources.resources
    assert r["denarii"].max_storage is None
    assert r["wheat"].max_storage == 2400
    assert r["wood"].max_storage == 3200
    assert r["marble"].max_storage == 3200


def test_passive_income(cfg: GameConfig) -> None:
    assert cfg.resources.passive_income.denarii == 25


# ---------------------------------------------------------------------------
# Bâtiments
# ---------------------------------------------------------------------------

def test_buildings_count(cfg: GameConfig) -> None:
    assert len(cfg.buildings.buildings) == 20


def test_building_ids(cfg: GameConfig) -> None:
    assert set(cfg.buildings.buildings.keys()) == EXPECTED_BUILDING_IDS


def test_building_sizes(cfg: GameConfig) -> None:
    b = cfg.buildings.buildings
    assert b["road"].size == (1, 1)
    assert b["housing"].size == (2, 2)
    assert b["forum"].size == (4, 4)
    assert b["wheat_farm"].size == (3, 3)
    assert b["obelisk"].size == (1, 1)


def test_building_costs(cfg: GameConfig) -> None:
    b = cfg.buildings.buildings
    assert b["temple"].cost == {"denarii": 800, "marble": 80}
    assert b["housing"].cost == {"wood": 10}
    assert b["forum"].cost == {"denarii": 1500, "marble": 150}
    assert b["obelisk"].cost == {"denarii": 1000, "marble": 500}


def test_building_maintenance(cfg: GameConfig) -> None:
    """Vérifie que maintenance == ceil(denarii_cost * 0.05) sauf road et housing (0)."""
    zero_maintenance = {"road", "housing"}
    for bld_id, bld in cfg.buildings.buildings.items():
        if bld_id in zero_maintenance:
            assert bld.maintenance == 0, f"{bld_id}: maintenance devrait être 0"
        else:
            denarii_cost = bld.cost.get("denarii", 0)
            expected = math.ceil(denarii_cost * 0.05)
            assert bld.maintenance == expected, (
                f"{bld_id}: maintenance={bld.maintenance}, attendu={expected} "
                f"(ceil({denarii_cost} * 0.05))"
            )


def test_unique_buildings(cfg: GameConfig) -> None:
    unique_ids = {bld_id for bld_id, bld in cfg.buildings.buildings.items() if bld.unique}
    assert unique_ids == {"forum", "obelisk"}


def test_terrain_constraints(cfg: GameConfig) -> None:
    b = cfg.buildings.buildings
    assert b["wheat_farm"].terrain_constraint is not None
    assert b["wheat_farm"].terrain_constraint.type == "all_tiles"
    assert b["wheat_farm"].terrain_constraint.terrain.value == "plain"

    assert b["lumber_camp"].terrain_constraint is not None
    assert b["lumber_camp"].terrain_constraint.type == "adjacent"
    assert b["lumber_camp"].terrain_constraint.terrain.value == "forest"

    assert b["marble_quarry"].terrain_constraint is not None
    assert b["marble_quarry"].terrain_constraint.type == "all_tiles"
    assert b["marble_quarry"].terrain_constraint.terrain.value == "hill"

    assert b["road"].terrain_constraint is None
    assert b["well"].terrain_constraint is None
    assert b["forum"].terrain_constraint is None


# ---------------------------------------------------------------------------
# Niveaux de maison
# ---------------------------------------------------------------------------

def test_house_levels_count(cfg: GameConfig) -> None:
    assert len(cfg.needs.house_levels) == 6


def test_house_levels_needs_cumulative(cfg: GameConfig) -> None:
    """Chaque niveau doit avoir les besoins du niveau précédent + au moins un de plus."""
    levels = sorted(cfg.needs.house_levels, key=lambda h: h.level)
    for i in range(1, len(levels)):
        prev_needs = set(levels[i - 1].required_needs)
        curr_needs = set(levels[i].required_needs)
        assert prev_needs.issubset(curr_needs), (
            f"Niveau {levels[i].level} ({levels[i].id}) ne contient pas tous les besoins "
            f"du niveau précédent. Manquants : {prev_needs - curr_needs}"
        )


def test_house_levels_tax_rates(cfg: GameConfig) -> None:
    levels = sorted(cfg.needs.house_levels, key=lambda h: h.level)
    expected_taxes = [0.2, 0.3, 0.3, 0.35, 0.4, 0.5]
    for level, expected in zip(levels, expected_taxes):
        assert level.tax_per_inhabitant == pytest.approx(expected), (
            f"Niveau {level.level} ({level.id}): taxe={level.tax_per_inhabitant}, attendu={expected}"
        )


def test_house_levels_max_populations(cfg: GameConfig) -> None:
    levels = sorted(cfg.needs.house_levels, key=lambda h: h.level)
    expected_pops = [5, 10, 20, 35, 50, 70]
    for level, expected in zip(levels, expected_pops):
        assert level.max_population == expected


# ---------------------------------------------------------------------------
# Événements
# ---------------------------------------------------------------------------

def test_events_count(cfg: GameConfig) -> None:
    assert len(cfg.events.events) == 4


def test_events_ids(cfg: GameConfig) -> None:
    assert set(cfg.events.events.keys()) == {"fire", "drought", "good_harvest", "immigration"}


def test_events_probabilities_sum(cfg: GameConfig) -> None:
    total = sum(e.probability for e in cfg.events.events.values())
    assert total == pytest.approx(0.14)
    assert total < 1.0


def test_drought_duration(cfg: GameConfig) -> None:
    assert cfg.events.events["drought"].duration == 3


def test_fire_has_prevention(cfg: GameConfig) -> None:
    fire = cfg.events.events["fire"]
    assert fire.prevention is not None
    assert fire.prevention.building == "prefecture"
    assert fire.prevention.risk_divisor == 2


# ---------------------------------------------------------------------------
# Niveaux de ville
# ---------------------------------------------------------------------------

def test_city_levels_count(cfg: GameConfig) -> None:
    assert len(cfg.city_levels.city_levels) == 5


def test_city_levels_ids(cfg: GameConfig) -> None:
    ids = {cl.id for cl in cfg.city_levels.city_levels}
    assert ids == {"camp", "village", "town", "city", "nova_roma"}


def test_city_levels_ordering(cfg: GameConfig) -> None:
    levels = sorted(cfg.city_levels.city_levels, key=lambda cl: cl.level)
    pops = [cl.min_population for cl in levels]
    assert pops == sorted(pops)


def test_city_levels_required_buildings_exist(cfg: GameConfig) -> None:
    building_ids = set(cfg.buildings.buildings.keys())
    for city_level in cfg.city_levels.city_levels:
        for bld_id in city_level.required_buildings:
            assert bld_id in building_ids, (
                f"Niveau '{city_level.id}' référence un bâtiment inconnu : '{bld_id}'"
            )


# ---------------------------------------------------------------------------
# Références croisées
# ---------------------------------------------------------------------------

def test_cross_references_storage_buildings(cfg: GameConfig) -> None:
    building_ids = set(cfg.buildings.buildings.keys())
    for res_id, res in cfg.resources.resources.items():
        if res.storage_building is not None:
            assert res.storage_building in building_ids, (
                f"Ressource '{res_id}' référence un bâtiment de stockage inconnu : "
                f"'{res.storage_building}'"
            )


def test_service_types_are_valid_needs(cfg: GameConfig) -> None:
    for bld_id, bld in cfg.buildings.buildings.items():
        if bld.service is not None:
            assert bld.service.type in VALID_NEEDS, (
                f"Bâtiment '{bld_id}' a un service de type inconnu : '{bld.service.type}'"
            )


# ---------------------------------------------------------------------------
# Gestion d'erreur
# ---------------------------------------------------------------------------

def test_invalid_config_dir_raises(tmp_path: pytest.TempPathFactory) -> None:
    """Un répertoire sans YAML doit lever FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config(config_dir=tmp_path)  # type: ignore[arg-type]
