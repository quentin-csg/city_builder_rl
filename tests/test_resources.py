"""Tests pour vitruvius.engine.resources (runtime)."""

import pytest

from vitruvius.config import load_config
from vitruvius.engine.grid import PlacedBuilding
from vitruvius.engine.resources import (
    ResourceState,
    apply_maintenance,
    apply_passive_income,
    apply_production,
    apply_taxes,
    apply_wheat_consumption,
    can_afford,
    compute_storage_cap,
    get_stock,
    init_resources,
    pay_cost,
    set_stock,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def bldg(cfg):
    return cfg.buildings.buildings


@pytest.fixture(scope="module")
def res_cfg(cfg):
    return cfg.resources


def _place(building_id: str, x: int = 0, y: int = 0, size: tuple[int, int] = (1, 1)) -> PlacedBuilding:
    """PlacedBuilding minimal pour les tests (pas de grille réelle nécessaire)."""
    return PlacedBuilding(building_id=building_id, x=x, y=y, size=size)


def _placed(*building_ids: str) -> dict[tuple[int, int], PlacedBuilding]:
    """Construit un dict placed_buildings avec des coordonnées artificielles."""
    return {(i, 0): _place(bid, x=i) for i, bid in enumerate(building_ids)}


# ---------------------------------------------------------------------------
# init_resources
# ---------------------------------------------------------------------------


def test_init_from_config(res_cfg):
    state = init_resources(res_cfg)
    assert state.denarii == 800.0
    assert state.wheat == 200
    assert state.wood == 100
    assert state.marble == 0


def test_init_types(res_cfg):
    state = init_resources(res_cfg)
    assert isinstance(state.denarii, float)
    assert isinstance(state.wheat, int)
    assert isinstance(state.wood, int)
    assert isinstance(state.marble, int)


# ---------------------------------------------------------------------------
# get_stock / set_stock
# ---------------------------------------------------------------------------


def test_get_stock_all_keys():
    state = ResourceState(denarii=100.0, wheat=50, wood=30, marble=10)
    assert get_stock(state, "denarii") == 100.0
    assert get_stock(state, "wheat") == 50
    assert get_stock(state, "wood") == 30
    assert get_stock(state, "marble") == 10


def test_set_stock_all_keys():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    set_stock(state, "denarii", 500.0)
    set_stock(state, "wheat", 100)
    set_stock(state, "wood", 200)
    set_stock(state, "marble", 50)
    assert state.denarii == 500.0
    assert state.wheat == 100
    assert state.wood == 200
    assert state.marble == 50


def test_get_stock_invalid_key():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    with pytest.raises(ValueError):
        get_stock(state, "gold")


def test_set_stock_invalid_key():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    with pytest.raises(ValueError):
        set_stock(state, "food", 10)


# ---------------------------------------------------------------------------
# compute_storage_cap
# ---------------------------------------------------------------------------


def test_cap_no_buildings(bldg):
    placed = {}
    assert compute_storage_cap("wheat", placed, bldg) == 0
    assert compute_storage_cap("wood", placed, bldg) == 0
    assert compute_storage_cap("marble", placed, bldg) == 0


def test_cap_one_granary(bldg):
    placed = _placed("granary")
    assert compute_storage_cap("wheat", placed, bldg) == 2400


def test_cap_two_granaries(bldg):
    placed = _placed("granary", "granary")
    assert compute_storage_cap("wheat", placed, bldg) == 4800


def test_cap_denarii_unlimited(bldg):
    placed = _placed("granary", "warehouse_wood")
    assert compute_storage_cap("denarii", placed, bldg) is None


def test_cap_mixed_buildings(bldg):
    # granary + warehouse_wood posés : wheat cap = 2400, wood cap = 3200
    placed = _placed("granary", "warehouse_wood")
    assert compute_storage_cap("wheat", placed, bldg) == 2400
    assert compute_storage_cap("wood", placed, bldg) == 3200
    assert compute_storage_cap("marble", placed, bldg) == 0


# ---------------------------------------------------------------------------
# apply_production
# ---------------------------------------------------------------------------


def test_prod_with_storage(bldg, res_cfg):
    state = ResourceState(denarii=800.0, wheat=0, wood=0, marble=0)
    placed = _placed("wheat_farm", "granary")
    result = apply_production(state, placed, bldg, res_cfg)
    assert state.wheat == 12
    assert result.get("wheat", 0) == 12


def test_prod_without_storage(bldg, res_cfg):
    state = ResourceState(denarii=800.0, wheat=100, wood=0, marble=0)
    placed = _placed("wheat_farm")  # pas de granary
    apply_production(state, placed, bldg, res_cfg)
    assert state.wheat == 100  # inchangé


def test_prod_capped(bldg, res_cfg):
    # Cap = 2400, wheat actuel = 2395, ferme produit 12 → seulement 5 ajoutés
    state = ResourceState(denarii=800.0, wheat=2395, wood=0, marble=0)
    placed = _placed("wheat_farm", "granary")
    result = apply_production(state, placed, bldg, res_cfg)
    assert state.wheat == 2400
    assert result.get("wheat", 0) == 5


def test_prod_denarii_no_cap(bldg, res_cfg):
    state = ResourceState(denarii=100.0, wheat=0, wood=0, marble=0)
    placed = _placed("trading_post")
    result = apply_production(state, placed, bldg, res_cfg)
    assert state.denarii == 150.0  # 100 + 50
    assert result.get("denarii", 0) == 50


def test_prod_multiple(bldg, res_cfg):
    # 2 wheat_farms + 1 granary → +24 blé
    state = ResourceState(denarii=800.0, wheat=0, wood=0, marble=0)
    placed = _placed("wheat_farm", "wheat_farm", "granary")
    result = apply_production(state, placed, bldg, res_cfg)
    assert state.wheat == 24
    assert result.get("wheat", 0) == 24


def test_prod_at_cap(bldg, res_cfg):
    # Déjà au cap → 0 produit
    state = ResourceState(denarii=800.0, wheat=2400, wood=0, marble=0)
    placed = _placed("wheat_farm", "granary")
    result = apply_production(state, placed, bldg, res_cfg)
    assert state.wheat == 2400
    assert result.get("wheat", 0) == 0


def test_prod_marble_quarry_with_warehouse(bldg, res_cfg):
    state = ResourceState(denarii=800.0, wheat=0, wood=0, marble=0)
    placed = _placed("marble_quarry", "warehouse_marble")
    result = apply_production(state, placed, bldg, res_cfg)
    assert state.marble == 8
    assert result.get("marble", 0) == 8


def test_prod_marble_quarry_without_warehouse(bldg, res_cfg):
    state = ResourceState(denarii=800.0, wheat=0, wood=0, marble=0)
    placed = _placed("marble_quarry")  # pas de warehouse_marble
    apply_production(state, placed, bldg, res_cfg)
    assert state.marble == 0


def test_prod_farm_modifier_positive(bldg, res_cfg):
    # wheat_farm produit 12/tour ; farm_modifier=+0.5 → floor(12 * 1.5) = 18
    state = ResourceState(denarii=800.0, wheat=0, wood=0, marble=0)
    placed = _placed("wheat_farm", "granary")
    result = apply_production(state, placed, bldg, res_cfg, farm_modifier=0.5)
    assert state.wheat == 18
    assert result.get("wheat", 0) == 18


def test_prod_farm_modifier_negative(bldg, res_cfg):
    # farm_modifier=-0.5 (sécheresse) → floor(12 * 0.5) = 6
    state = ResourceState(denarii=800.0, wheat=0, wood=0, marble=0)
    placed = _placed("wheat_farm", "granary")
    result = apply_production(state, placed, bldg, res_cfg, farm_modifier=-0.5)
    assert state.wheat == 6
    assert result.get("wheat", 0) == 6


def test_prod_farm_modifier_no_effect_on_denarii(bldg, res_cfg):
    # farm_modifier ne touche pas denarii (trading_post)
    state = ResourceState(denarii=100.0, wheat=0, wood=0, marble=0)
    placed = _placed("trading_post")
    result = apply_production(state, placed, bldg, res_cfg, farm_modifier=0.5)
    assert state.denarii == 150.0  # 100 + 50, inchangé par le modifier
    assert result.get("denarii", 0) == 50


# ---------------------------------------------------------------------------
# apply_passive_income
# ---------------------------------------------------------------------------


def test_passive_income(res_cfg):
    state = ResourceState(denarii=100.0, wheat=0, wood=0, marble=0)
    added = apply_passive_income(state, res_cfg.passive_income)
    assert state.denarii == 140.0
    assert added == 40.0


# ---------------------------------------------------------------------------
# apply_maintenance
# ---------------------------------------------------------------------------


def test_maint_single(bldg):
    state = ResourceState(denarii=100.0, wheat=0, wood=0, marble=0)
    placed = _placed("well")  # maintenance = 3
    cost = apply_maintenance(state, placed, bldg)
    assert state.denarii == 97.0
    assert cost == 3.0


def test_maint_multiple(bldg):
    state = ResourceState(denarii=200.0, wheat=0, wood=0, marble=0)
    # well(3) + aqueduct(2) + wheat_farm(10) = 15
    placed = _placed("well", "aqueduct", "wheat_farm")
    cost = apply_maintenance(state, placed, bldg)
    assert state.denarii == 185.0
    assert cost == 15.0


def test_maint_road_zero(bldg):
    state = ResourceState(denarii=50.0, wheat=0, wood=0, marble=0)
    placed = _placed("road")  # maintenance = 0
    cost = apply_maintenance(state, placed, bldg)
    assert state.denarii == 50.0
    assert cost == 0.0


def test_maint_housing_zero(bldg):
    state = ResourceState(denarii=50.0, wheat=0, wood=0, marble=0)
    placed = _placed("housing")  # maintenance = 0
    cost = apply_maintenance(state, placed, bldg)
    assert state.denarii == 50.0
    assert cost == 0.0


def test_maint_negative(bldg):
    state = ResourceState(denarii=5.0, wheat=0, wood=0, marble=0)
    placed = _placed("forum")  # maintenance = 40
    apply_maintenance(state, placed, bldg)
    assert state.denarii == -35.0


def test_maint_returns_total(bldg):
    state = ResourceState(denarii=1000.0, wheat=0, wood=0, marble=0)
    placed = _placed("well", "well")  # 2 × 3 = 6
    cost = apply_maintenance(state, placed, bldg)
    assert cost == 6.0


# ---------------------------------------------------------------------------
# apply_taxes
# ---------------------------------------------------------------------------


def test_taxes_empty():
    state = ResourceState(denarii=100.0, wheat=0, wood=0, marble=0)
    total = apply_taxes(state, [])
    assert state.denarii == 100.0
    assert total == 0.0


def test_taxes_multiple():
    state = ResourceState(denarii=50.0, wheat=0, wood=0, marble=0)
    total = apply_taxes(state, [10.0, 15.0, 7.0])
    assert state.denarii == 82.0
    assert total == 32.0


def test_taxes_returns_total():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    total = apply_taxes(state, [5.0, 5.0])
    assert total == 10.0


# ---------------------------------------------------------------------------
# apply_wheat_consumption
# ---------------------------------------------------------------------------


def test_conso_sufficient():
    # 3 maisons, blé largement suffisant
    state = ResourceState(denarii=0.0, wheat=100, wood=0, marble=0)
    flags = apply_wheat_consumption(state, [10, 20, 5])
    # needs: ceil(10/10)=1, ceil(20/10)=2, ceil(5/10)=1 → total=4
    assert flags == [False, False, False]
    assert state.wheat == 96


def test_conso_fifo_famine():
    # wheat=3, maisons [10, 20, 30] → needs [1, 2, 3]
    # house1: 3>=1 → feed, wheat=2
    # house2: 2>=2 → feed, wheat=0
    # house3: 0<3 → famine, wheat=0
    state = ResourceState(denarii=0.0, wheat=3, wood=0, marble=0)
    flags = apply_wheat_consumption(state, [10, 20, 30])
    assert flags == [False, False, True]
    assert state.wheat == 0


def test_conso_zero_pop():
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    flags = apply_wheat_consumption(state, [0, 0, 0])
    assert flags == [False, False, False]
    assert state.wheat == 0


def test_conso_ceil_rounding():
    # pop=1 → ceil(1/10)=1, pop=11 → ceil(11/10)=2
    state = ResourceState(denarii=0.0, wheat=10, wood=0, marble=0)
    flags = apply_wheat_consumption(state, [1, 11])
    assert flags == [False, False]
    assert state.wheat == 7  # 10 - 1 - 2


def test_conso_no_wheat():
    # Toutes en famine sauf pop=0
    state = ResourceState(denarii=0.0, wheat=0, wood=0, marble=0)
    flags = apply_wheat_consumption(state, [0, 10, 20])
    assert flags == [False, True, True]


def test_conso_exact_wheat():
    # Exactement le bon montant
    state = ResourceState(denarii=0.0, wheat=3, wood=0, marble=0)
    # needs: 1+2 = 3
    flags = apply_wheat_consumption(state, [10, 20])
    assert flags == [False, False]
    assert state.wheat == 0


def test_conso_fifo_smaller_house_after_famine():
    # wheat=1, maisons [20, 10] → needs [2, 1]
    # house1: 1<2 → famine, wheat reste à 1
    # house2: 1>=1 → feed, wheat=0
    state = ResourceState(denarii=0.0, wheat=1, wood=0, marble=0)
    flags = apply_wheat_consumption(state, [20, 10])
    assert flags == [True, False]
    assert state.wheat == 0


# ---------------------------------------------------------------------------
# can_afford / pay_cost
# ---------------------------------------------------------------------------


def test_can_afford_true():
    state = ResourceState(denarii=500.0, wheat=100, wood=50, marble=20)
    assert can_afford(state, {"denarii": 200, "wood": 30})


def test_can_afford_false():
    state = ResourceState(denarii=100.0, wheat=50, wood=10, marble=0)
    assert not can_afford(state, {"denarii": 200})  # insuffisant


def test_can_afford_exact():
    state = ResourceState(denarii=200.0, wheat=0, wood=0, marble=0)
    assert can_afford(state, {"denarii": 200})


def test_pay_cost():
    state = ResourceState(denarii=800.0, wheat=200, wood=100, marble=50)
    pay_cost(state, {"denarii": 300, "marble": 50})
    assert state.denarii == 500.0
    assert state.marble == 0
    assert state.wheat == 200  # inchangé
    assert state.wood == 100  # inchangé
