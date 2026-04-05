"""Tests pour vitruvius.engine.events."""

import math

import numpy as np
import pytest

from vitruvius.config import load_config
from vitruvius.engine.buildings import try_place_building
from vitruvius.engine.events import (
    ActiveEvent,
    apply_event,
    draw_event,
    get_farm_modifier,
    process_events,
    tick_events,
)
from vitruvius.engine.grid import Grid
from vitruvius.engine.population import HouseState, apply_immigration
from vitruvius.engine.resources import ResourceState
from vitruvius.engine.terrain import TerrainType


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
def events_cfg(cfg):
    return cfg.events.events


@pytest.fixture(scope="module")
def house_levels(cfg):
    return cfg.needs.house_levels


@pytest.fixture
def grid():
    return Grid(seed=42)


@pytest.fixture
def state():
    return ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rng_with_value(r: float) -> np.random.Generator:
    """Crée un RNG dont le premier appel à random() retourne r (via mock-like)."""
    # On utilise un RNG patché : on cherche un seed qui donne un r dans [target, target+0.001)
    # Mais plus simple : on injecte directement via numpy avec un mock minimal.
    class _FixedRNG:
        """RNG minimal retournant une valeur fixe au premier random()."""

        def __init__(self, value: float, seed: int = 42):
            self._value = value
            self._delegate = np.random.default_rng(seed)
            self._used = False

        def random(self) -> float:
            if not self._used:
                self._used = True
                return self._value
            return self._delegate.random()

        def integers(self, low: int, high: int) -> int:
            return int(self._delegate.integers(low, high))

    return _FixedRNG(r)  # type: ignore[return-value]


def _plain_tiles(grid: Grid) -> list[tuple[int, int]]:
    return [
        (x, y)
        for y in range(grid.SIZE)
        for x in range(grid.SIZE)
        if grid.terrain[y][x] == TerrainType.PLAIN and grid._origin[y][x] is None
    ]


def _place(grid: Grid, state: ResourceState, bldg, building_id: str) -> tuple[int, int] | None:
    for x, y in _plain_tiles(grid):
        if try_place_building(grid, state, building_id, x, y, bldg):
            return (x, y)
    return None


# ---------------------------------------------------------------------------
# draw_event
# ---------------------------------------------------------------------------


def test_draw_fire(events_cfg):
    rng = _make_rng_with_value(0.01)
    assert draw_event(events_cfg, rng) == "fire"


def test_draw_drought(events_cfg):
    rng = _make_rng_with_value(0.04)
    assert draw_event(events_cfg, rng) == "drought"


def test_draw_good_harvest(events_cfg):
    rng = _make_rng_with_value(0.07)
    assert draw_event(events_cfg, rng) == "good_harvest"


def test_draw_immigration(events_cfg):
    rng = _make_rng_with_value(0.12)
    assert draw_event(events_cfg, rng) == "immigration"


def test_draw_no_event(events_cfg):
    rng = _make_rng_with_value(0.50)
    assert draw_event(events_cfg, rng) is None


def test_draw_boundary_fire_drought(events_cfg):
    # r=0.03 exact : cumulative apres fire = 0.03, donc r < 0.03 est False -> drought
    rng = _make_rng_with_value(0.03)
    assert draw_event(events_cfg, rng) == "drought"


def test_draw_boundary_drought_harvest(events_cfg):
    # cumulative apres drought = 0.05 -> r=0.05 -> good_harvest
    rng = _make_rng_with_value(0.05)
    assert draw_event(events_cfg, rng) == "good_harvest"


def test_draw_boundary_harvest_immigration(events_cfg):
    # cumulative apres good_harvest = 0.10 -> r=0.10 -> immigration
    rng = _make_rng_with_value(0.10)
    assert draw_event(events_cfg, rng) == "immigration"


def test_draw_boundary_immigration_none(events_cfg):
    # cumulative apres immigration = 0.14 -> r=0.14 -> None
    rng = _make_rng_with_value(0.14)
    assert draw_event(events_cfg, rng) is None


def test_draw_deterministic_seed(events_cfg):
    rng1 = np.random.default_rng(0)
    rng2 = np.random.default_rng(0)
    results1 = [draw_event(events_cfg, rng1) for _ in range(20)]
    results2 = [draw_event(events_cfg, rng2) for _ in range(20)]
    assert results1 == results2


# ---------------------------------------------------------------------------
# fire
# ---------------------------------------------------------------------------


def test_fire_destroys_building(events_cfg, bldg, grid, state):
    origin = _place(grid, state, bldg, "well")
    assert origin is not None
    count_before = len(grid.placed_buildings)
    rng = np.random.default_rng(0)
    event = apply_event("fire", events_cfg["fire"], [], rng, grid, state, bldg)
    assert event is not None
    assert event.event_type == "fire"
    assert len(grid.placed_buildings) == count_before - 1


def test_fire_no_refund(events_cfg, bldg):
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=500)
    _place(g, s, bldg, "well")
    denarii_before = s.denarii
    rng = np.random.default_rng(0)
    apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    # Le remboursement de try_demolish n'a PAS lieu : denarii ne doit pas augmenter
    # (le well coûte du denarii, donc si remboursement il y aurait eu)
    assert s.denarii == denarii_before


def test_fire_immune_unique(events_cfg, bldg):
    """Forum et obélisque ne doivent jamais être détruits."""
    g = Grid(seed=42)
    s = ResourceState(denarii=10000.0, wheat=0, wood=1000, marble=1000)
    # Placer uniquement des bâtiments uniques
    forum_placed = _place(g, s, bldg, "forum")
    obelisk_placed = _place(g, s, bldg, "obelisk")
    assert forum_placed is not None or obelisk_placed is not None
    # Supprimer les non-uniques qui auraient pu être posés par fixture de grid
    # (ici grid est neuf, donc seulement forum/obelisque)
    rng = np.random.default_rng(0)
    event = apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    # Aucun batiment eligible -> fizzle
    assert event is None


def test_fire_no_eligible_buildings(events_cfg, bldg):
    g = Grid(seed=42)
    s = ResourceState(denarii=10000.0, wheat=0, wood=1000, marble=1000)
    _place(g, s, bldg, "forum")
    _place(g, s, bldg, "obelisk")
    # Seuls des uniques sur la grille
    rng = np.random.default_rng(0)
    event = apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    assert event is None


def test_fire_empty_grid(events_cfg, bldg):
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=500)
    rng = np.random.default_rng(0)
    event = apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    assert event is None


def test_fire_prefecture_fizzle(events_cfg, bldg):
    """Quand la prefecture couvre la cible et le second roll cause un fizzle."""
    g = Grid(seed=42)
    s = ResourceState(denarii=10000.0, wheat=0, wood=1000, marble=1000)
    farm_origin = _place(g, s, bldg, "well")
    assert farm_origin is not None

    # Placer prefecture adjacente pour couvrir la ferme
    # La prefecture a radius=20, donc elle couvre quasiment tout
    _place(g, s, bldg, "prefecture")

    # RNG : premier appel random() -> < 0.03 (fire tiré), second -> 0.0 (< 0.5 = fizzle)
    class _FizzleRNG:
        def __init__(self):
            self._calls = 0
            self._delegate = np.random.default_rng(42)

        def random(self) -> float:
            self._calls += 1
            if self._calls == 1:
                return 0.0  # fizzle (< 0.5)
            return 0.0

        def integers(self, low: int, high: int) -> int:
            return 0  # selectionne toujours le premier batiment eligible

    rng = _FizzleRNG()  # type: ignore[assignment]
    event = apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    assert event is None


def test_fire_prefecture_no_fizzle(events_cfg, bldg):
    """Quand la prefecture couvre mais le second roll passe -> destruction."""
    g = Grid(seed=42)
    s = ResourceState(denarii=10000.0, wheat=0, wood=1000, marble=1000)
    _place(g, s, bldg, "well")
    _place(g, s, bldg, "prefecture")
    count_before = len(g.placed_buildings)

    class _NoFizzleRNG:
        def __init__(self):
            self._calls = 0
            self._delegate = np.random.default_rng(42)

        def random(self) -> float:
            self._calls += 1
            if self._calls == 1:
                return 0.99  # ne fizzle pas (>= 0.5)
            return 0.99

        def integers(self, low: int, high: int) -> int:
            return 0

    rng = _NoFizzleRNG()  # type: ignore[assignment]
    event = apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    assert event is not None
    assert len(g.placed_buildings) == count_before - 1


def test_fire_no_prefecture_no_second_roll(events_cfg, bldg):
    """Sans prefecture, pas de second roll : le batiment brule directement."""
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=500)
    _place(g, s, bldg, "well")
    count_before = len(g.placed_buildings)
    rng = np.random.default_rng(99)
    event = apply_event("fire", events_cfg["fire"], [], rng, g, s, bldg)
    assert event is not None
    assert len(g.placed_buildings) == count_before - 1


def test_fire_clamps_storage(events_cfg, bldg):
    """Détruire un grenier clamp le stock de blé."""
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=0, wood=500, marble=500)
    granary_origin = _place(g, s, bldg, "granary")
    assert granary_origin is not None

    from vitruvius.engine.resources import compute_storage_cap
    cap = compute_storage_cap("wheat", g.placed_buildings, bldg)
    s.wheat = cap  # Remplir le grenier

    # Choisir le grenier comme cible
    from vitruvius.engine.events import _apply_fire

    class _TargetGranaryRNG:
        def __init__(self, target_idx: int):
            self._target_idx = target_idx
            self._delegate = np.random.default_rng(42)

        def random(self) -> float:
            return self._delegate.random()

        def integers(self, low: int, high: int) -> int:
            # Toujours retourner l'index du grenier
            eligible_ids = [
                pb.building_id
                for pb in g.placed_buildings.values()
                if not bldg[pb.building_id].unique
            ]
            try:
                idx = eligible_ids.index("granary")
            except ValueError:
                idx = 0
            return idx

    rng = _TargetGranaryRNG(0)  # type: ignore[assignment]
    event = _apply_fire(events_cfg["fire"], rng, g, s, bldg)  # type: ignore[arg-type]
    # Le grenier est detruit, cap -> 0, wheat clampé a 0
    assert s.wheat == 0


# ---------------------------------------------------------------------------
# drought
# ---------------------------------------------------------------------------


def test_drought_creates_active_event(events_cfg):
    active: list[ActiveEvent] = []
    event = apply_event("drought", events_cfg["drought"], active, np.random.default_rng(0),
                        Grid(seed=42), ResourceState(5000.0, 0, 0, 0), {})
    assert event is not None
    assert event.event_type == "drought"
    assert event.turns_remaining == 3
    assert event.data["modifier"] == pytest.approx(-0.5)
    assert event in active


def test_drought_stacking_resets_timer(events_cfg):
    """Sécheresse déjà active : reset timer, pas de second modifier."""
    active: list[ActiveEvent] = [ActiveEvent("drought", 1, {"modifier": -0.5})]
    event = apply_event("drought", events_cfg["drought"], active, np.random.default_rng(0),
                        Grid(seed=42), ResourceState(5000.0, 0, 0, 0), {})
    assert len(active) == 1  # pas de second evenement
    assert active[0].turns_remaining == 3  # timer reset
    assert event is active[0]


# ---------------------------------------------------------------------------
# good_harvest
# ---------------------------------------------------------------------------


def test_good_harvest_creates_active_event(events_cfg):
    active: list[ActiveEvent] = []
    event = apply_event("good_harvest", events_cfg["good_harvest"], active, np.random.default_rng(0),
                        Grid(seed=42), ResourceState(5000.0, 0, 0, 0), {})
    assert event is not None
    assert event.event_type == "good_harvest"
    assert event.turns_remaining == 1
    assert event.data["modifier"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# get_farm_modifier
# ---------------------------------------------------------------------------


def test_get_farm_modifier_drought():
    active = [ActiveEvent("drought", 2, {"modifier": -0.5})]
    assert get_farm_modifier(active) == pytest.approx(-0.5)


def test_get_farm_modifier_good_harvest():
    active = [ActiveEvent("good_harvest", 1, {"modifier": 0.5})]
    assert get_farm_modifier(active) == pytest.approx(0.5)


def test_get_farm_modifier_none():
    assert get_farm_modifier([]) == pytest.approx(0.0)


def test_get_farm_modifier_combined():
    active = [
        ActiveEvent("drought", 2, {"modifier": -0.5}),
        ActiveEvent("good_harvest", 1, {"modifier": 0.5}),
    ]
    assert get_farm_modifier(active) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# immigration
# ---------------------------------------------------------------------------


def test_immigration_adds_population(events_cfg, bldg, house_levels):
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    # Placer quelques housings
    for _ in range(3):
        _place(g, s, bldg, "housing")
    from vitruvius.engine.population import init_houses
    houses = init_houses(g, bldg)
    active: list[ActiveEvent] = []
    rng = np.random.default_rng(0)
    event = apply_event("immigration", events_cfg["immigration"], active, rng,
                        g, s, bldg, houses, house_levels)
    assert event is not None
    assert event.event_type == "immigration"
    assert event.data["amount"] >= 50
    assert event.data["amount"] <= 150
    assert event.data["settled"] > 0
    total_pop = sum(h.population for h in houses.values())
    assert total_pop == event.data["settled"]


def test_immigration_no_houses(events_cfg, bldg, house_levels):
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    active: list[ActiveEvent] = []
    rng = np.random.default_rng(0)
    event = apply_event("immigration", events_cfg["immigration"], active, rng,
                        g, s, bldg, {}, house_levels)
    assert event is not None
    assert event.data["settled"] == 0


def test_immigration_amount_in_range(events_cfg, bldg, house_levels):
    """Le montant reste dans [50, 150] sur plusieurs tirages."""
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    for seed in range(10):
        active: list[ActiveEvent] = []
        rng = np.random.default_rng(seed)
        event = apply_event("immigration", events_cfg["immigration"], active, rng,
                            g, s, bldg, {}, house_levels)
        assert event is not None
        assert 50 <= event.data["amount"] <= 150


# ---------------------------------------------------------------------------
# tick_events
# ---------------------------------------------------------------------------


def test_tick_decrements():
    events = [ActiveEvent("drought", 3, {}), ActiveEvent("fire", 1, {})]
    tick_events(events)
    assert len(events) == 1           # fire (tr=1->0) supprime
    assert events[0].turns_remaining == 2  # drought (tr=3->2) survit


def test_tick_removes_expired():
    events = [ActiveEvent("drought", 1, {})]
    tick_events(events)
    assert len(events) == 0


def test_tick_preserves_active():
    events = [ActiveEvent("drought", 5, {})]
    tick_events(events)
    assert len(events) == 1
    assert events[0].turns_remaining == 4


def test_tick_empty_list():
    events: list[ActiveEvent] = []
    tick_events(events)
    assert events == []


# ---------------------------------------------------------------------------
# process_events
# ---------------------------------------------------------------------------


def test_process_events_tick_before_draw(events_cfg, bldg):
    """tick se passe avant le draw : un evenement existant à tr=1 est supprimé avant tirage."""
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    existing = ActiveEvent("drought", 1, {"modifier": -0.5})
    active = [existing]
    # RNG : r=0.50 -> pas de nouvel evenement
    rng = _make_rng_with_value(0.50)
    result = process_events(events_cfg, active, rng, g, s, bldg)  # type: ignore[arg-type]
    assert result is None
    # L'existant a ete tick (tr 1->0) puis supprime
    assert len(active) == 0


def test_process_events_new_event_created(events_cfg, bldg, house_levels):
    """Un nouvel événement drought est crée et existe dans active_events."""
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    active: list[ActiveEvent] = []
    rng = _make_rng_with_value(0.04)  # drought
    result = process_events(events_cfg, active, rng, g, s, bldg,  # type: ignore[arg-type]
                            None, None)
    assert result is not None
    assert result.event_type == "drought"
    assert result in active
    assert result.turns_remaining == 3


def test_process_events_no_event_existing_ticked(events_cfg, bldg):
    """Quand aucun evenement n'est tiré, les existants sont quand même tickés."""
    g = Grid(seed=42)
    s = ResourceState(denarii=5000.0, wheat=200, wood=500, marble=500)
    active = [ActiveEvent("drought", 3, {"modifier": -0.5})]
    rng = _make_rng_with_value(0.50)  # pas d'evenement
    result = process_events(events_cfg, active, rng, g, s, bldg)  # type: ignore[arg-type]
    assert result is None
    assert len(active) == 1
    assert active[0].turns_remaining == 2
