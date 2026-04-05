"""État complet du jeu, sérialisable en JSON."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from vitruvius.engine.events import ActiveEvent
from vitruvius.engine.grid import Grid
from vitruvius.engine.population import HouseState
from vitruvius.engine.resources import ResourceState, init_resources

if TYPE_CHECKING:
    from vitruvius.config import GameConfig


# ---------------------------------------------------------------------------
# Dataclass principale
# ---------------------------------------------------------------------------


@dataclass
class GameState:
    """État complet et mutable d'une partie.

    Unique objet passé entre chaque étape de la boucle de tour.
    Sérialisable en JSON via to_dict / from_dict.
    Déterministe : le RNG est seedé et son état est sérialisable.
    """

    grid: Grid
    resource_state: ResourceState
    houses: dict[tuple[int, int], HouseState]
    active_events: list[ActiveEvent]
    city_level: int                    # 1–5
    turn: int                          # incrémenté en fin de tour
    consecutive_bankrupt_turns: int    # tours consécutifs avec denarii < -500
    global_satisfaction: float         # 0.5 par convention si pop=0
    rng: np.random.Generator
    seed: int
    done: bool
    victory: bool


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def init_game_state(config: GameConfig, seed: int) -> GameState:
    """Crée un GameState initial à partir de la config et d'un seed.

    Args:
        config: Configuration unifiée du jeu.
        seed: Seed pour le RNG et la génération du terrain.

    Returns:
        GameState initialisé, prêt pour le premier tour.
    """
    grid = Grid(seed=seed)
    resource_state = init_resources(config.resources)
    rng = np.random.default_rng(seed)

    return GameState(
        grid=grid,
        resource_state=resource_state,
        houses={},
        active_events=[],
        city_level=1,
        turn=0,
        consecutive_bankrupt_turns=0,
        global_satisfaction=0.5,
        rng=rng,
        seed=seed,
        done=False,
        victory=False,
    )


# ---------------------------------------------------------------------------
# Sérialisation JSON
# ---------------------------------------------------------------------------


def to_dict(state: GameState) -> dict[str, Any]:
    """Convertit un GameState en dict JSON-sérialisable.

    Les tuples sont convertis en listes. Le RNG PCG64 est capturé via
    bit_generator.state. Le terrain de la grille est recalculé depuis le seed
    à la désérialisation — seuls les bâtiments posés sont stockés.

    Args:
        state: GameState à sérialiser.

    Returns:
        Dict contenant uniquement des types JSON-natifs (str, int, float, list, dict, bool).
    """
    # Ressources
    rs = state.resource_state
    resources_dict: dict[str, Any] = {
        "denarii": rs.denarii,
        "wheat": rs.wheat,
        "wood": rs.wood,
        "marble": rs.marble,
    }

    # Bâtiments posés (terrain reconstruit depuis seed)
    placed_list = [
        {"building_id": pb.building_id, "x": pb.x, "y": pb.y}
        for pb in state.grid.placed_buildings.values()
    ]

    # Maisons
    houses_list = [
        {
            "origin": list(h.origin),
            "level": h.level,
            "population": h.population,
            "famine": h.famine,
        }
        for h in state.houses.values()
    ]

    # Événements actifs
    def _serialize_event_data(edata: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in edata.items():
            out[k] = list(v) if isinstance(v, tuple) else v
        return out

    events_list = [
        {
            "event_type": e.event_type,
            "turns_remaining": e.turns_remaining,
            "data": _serialize_event_data(e.data),
        }
        for e in state.active_events
    ]

    # État RNG (PCG64 : ints natifs, JSON-compatible)
    rng_state: dict[str, Any] = state.rng.bit_generator.state

    return {
        "seed": state.seed,
        "turn": state.turn,
        "city_level": state.city_level,
        "consecutive_bankrupt_turns": state.consecutive_bankrupt_turns,
        "global_satisfaction": state.global_satisfaction,
        "done": state.done,
        "victory": state.victory,
        "resource_state": resources_dict,
        "placed_buildings": placed_list,
        "houses": houses_list,
        "active_events": events_list,
        "rng_state": rng_state,
    }


def from_dict(data: dict[str, Any], config: GameConfig) -> GameState:
    """Reconstruit un GameState depuis un dict sérialisé.

    Args:
        data: Dict produit par to_dict().
        config: Configuration unifiée du jeu (nécessaire pour replay des placements).

    Returns:
        GameState reconstruit, fonctionnellement équivalent à l'état d'origine.
    """
    seed: int = data["seed"]
    bldg = config.buildings.buildings

    # Grille : regénérer le terrain depuis le seed, puis rejouer les placements
    grid = Grid(seed=seed)
    for pb_data in data["placed_buildings"]:
        bid = pb_data["building_id"]
        x, y = pb_data["x"], pb_data["y"]
        grid.place_building(bid, x, y, bldg[bid])

    # Ressources
    rs_data = data["resource_state"]
    resource_state = ResourceState(
        denarii=rs_data["denarii"],
        wheat=rs_data["wheat"],
        wood=rs_data["wood"],
        marble=rs_data["marble"],
    )

    # Maisons (ordre d'insertion préservé)
    houses: dict[tuple[int, int], HouseState] = {}
    for h_data in data["houses"]:
        ox, oy = h_data["origin"]
        origin = (ox, oy)
        houses[origin] = HouseState(
            origin=origin,
            level=h_data["level"],
            population=h_data["population"],
            famine=h_data["famine"],
        )

    # Événements actifs
    def _deserialize_event_data(edata: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for k, v in edata.items():
            # Les tuples (destroyed_at) ont été convertis en listes
            if k == "destroyed_at" and isinstance(v, list):
                out[k] = tuple(v)
            else:
                out[k] = v
        return out

    active_events: list[ActiveEvent] = [
        ActiveEvent(
            event_type=e["event_type"],
            turns_remaining=e["turns_remaining"],
            data=_deserialize_event_data(e["data"]),
        )
        for e in data["active_events"]
    ]

    # RNG : restaurer l'état exact du générateur
    rng = np.random.default_rng(seed)
    rng.bit_generator.state = data["rng_state"]

    return GameState(
        grid=grid,
        resource_state=resource_state,
        houses=houses,
        active_events=active_events,
        city_level=data["city_level"],
        turn=data["turn"],
        consecutive_bankrupt_turns=data["consecutive_bankrupt_turns"],
        global_satisfaction=data["global_satisfaction"],
        rng=rng,
        seed=seed,
        done=data["done"],
        victory=data["victory"],
    )
