"""Encodage et décodage des messages JSON du protocole WebSocket.

Ce module est pur (pas d'I/O, pas d'asyncio) et entièrement testable de façon
synchrone. Toute validation du format client → serveur passe par ici.

Messages serveur → client :
    init  — envoyé à la connexion et après reset
    state — après chaque action (humaine ou auto)
    ack   — après load_model
    error — sur toute erreur client (connexion maintenue)

Messages client → serveur :
    action      — placer / démolir / do_nothing
    reset       — nouvelle partie (seed optionnel)
    load_model  — charger un modèle MaskablePPO (.zip)
    auto_step   — avancer de n tours avec le modèle chargé
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vitruvius.engine.turn import Action

if TYPE_CHECKING:
    from vitruvius.config import GameConfig
    from vitruvius.engine.game_state import GameState
    from vitruvius.engine.grid import Grid
    from vitruvius.engine.turn import TurnResult


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

AUTO_STEP_MAX: int = 1000


# ---------------------------------------------------------------------------
# Erreur protocole
# ---------------------------------------------------------------------------


class ProtocolError(ValueError):
    """Levée quand un message client est invalide.

    Contient un message lisible renvoyé directement au client.
    """


# ---------------------------------------------------------------------------
# Messages client (structurés)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionMsg:
    """Message client de type 'action'."""

    action: Action


@dataclass(frozen=True)
class ResetMsg:
    """Message client de type 'reset'."""

    seed: int | None


@dataclass(frozen=True)
class LoadModelMsg:
    """Message client de type 'load_model'."""

    path: str


@dataclass(frozen=True)
class AutoStepMsg:
    """Message client de type 'auto_step'."""

    n: int


ClientMessage = ActionMsg | ResetMsg | LoadModelMsg | AutoStepMsg


# ---------------------------------------------------------------------------
# Helpers terrain / catalog
# ---------------------------------------------------------------------------


def terrain_to_json(grid: Grid) -> list[list[str]]:
    """Convertit la grille de terrain en liste 2D de strings.

    Args:
        grid: Grille du jeu (32×32).

    Returns:
        Liste [y][x] de chaînes correspondant à TerrainType.value.
        Ex : "plain", "forest", "hill", "water", "marsh".
    """
    return [
        [grid.terrain[y][x].value for x in range(grid.SIZE)]
        for y in range(grid.SIZE)
    ]


def buildings_catalog_to_json(config: GameConfig) -> dict[str, Any]:
    """Exporte le catalogue des bâtiments pour Godot.

    Inclut uniquement les données utiles au rendu et aux tooltips : taille,
    coût, production, service, effets spéciaux, unicité. Les formules
    internes (satisfaction, taxes) ne sont pas exposées.

    Args:
        config: Configuration unifiée du jeu.

    Returns:
        Dict building_id → infos nécessaires côté client.
    """
    catalog: dict[str, Any] = {}
    for bid, cfg in config.buildings.buildings.items():
        entry: dict[str, Any] = {
            "display_name": cfg.display_name,
            "size": list(cfg.size),
            "cost": dict(cfg.cost),
            "unique": cfg.unique,
        }
        if cfg.production is not None:
            entry["production"] = {
                "resource": cfg.production.resource,
                "amount": cfg.production.amount,
            }
        if cfg.service is not None:
            entry["service"] = {
                "type": cfg.service.type,
                "radius": cfg.service.radius,
            }
        if cfg.special_effect is not None:
            se = cfg.special_effect
            entry["special_effect"] = {
                "is_housing": se.is_housing,
                "tax_bonus": se.tax_bonus,
                "requires_aqueduct": se.requires_aqueduct,
            }
        catalog[bid] = entry
    return catalog


# ---------------------------------------------------------------------------
# Sérialisation TurnResult
# ---------------------------------------------------------------------------


def _turn_result_to_json(result: TurnResult) -> dict[str, Any]:
    """Convertit TurnResult en dict JSON-sérialisable."""
    return {
        "production": result.production,
        "taxes_collected": result.taxes_collected,
        "maintenance_paid": result.maintenance_paid,
        "passive_income": result.passive_income,
        "famine_count": result.famine_count,
        "famine_pop_lost": result.famine_pop_lost,
        "evolved": result.evolved,
        "regressed": result.regressed,
        "growth": result.growth,
        "exodus": result.exodus,
        "new_event": result.new_event.event_type if result.new_event is not None else None,
        "global_satisfaction": result.global_satisfaction,
        "total_population": result.total_population,
        "city_level": result.city_level,
        "done": result.done,
        "victory": result.victory,
        "defeat": result.defeat,
        "bankrupt": result.bankrupt,
        "action_succeeded": result.action_succeeded,
    }


# ---------------------------------------------------------------------------
# Constructeurs de messages serveur
# ---------------------------------------------------------------------------


def build_init_message(gs: GameState, config: GameConfig, model_loaded: bool = False) -> dict[str, Any]:
    """Construit le message 'init' envoyé à la connexion et après reset.

    Args:
        gs: État initial du jeu.
        config: Configuration du jeu.
        model_loaded: True si un modèle est chargé dans la session.

    Returns:
        Dict JSON-sérialisable de type 'init'.
    """
    from vitruvius.engine.game_state import to_dict

    return {
        "type": "init",
        "seed": gs.seed,
        "size": gs.grid.SIZE,
        "terrain": terrain_to_json(gs.grid),
        "buildings_catalog": buildings_catalog_to_json(config),
        "state": to_dict(gs),
        "model_loaded": model_loaded,
    }


def build_state_message(gs: GameState, result: TurnResult) -> dict[str, Any]:
    """Construit le message 'state' envoyé après chaque action.

    Args:
        gs: État courant du jeu (après application de l'action).
        result: Résultat du tour.

    Returns:
        Dict JSON-sérialisable de type 'state'.
    """
    from vitruvius.engine.game_state import to_dict

    return {
        "type": "state",
        "state": to_dict(gs),
        "turn_result": _turn_result_to_json(result),
        "done": gs.done,
        "victory": gs.victory,
    }


def build_error_message(message: str) -> dict[str, Any]:
    """Construit un message 'error'.

    Args:
        message: Description de l'erreur, lisible par le client.

    Returns:
        Dict JSON-sérialisable de type 'error'.
    """
    return {"type": "error", "message": message}


def build_ack_message(action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Construit un message d'accusé de réception.

    Args:
        action: Nom de l'action acquittée (ex : "load_model").
        payload: Données complémentaires optionnelles.

    Returns:
        Dict JSON-sérialisable de type 'ack'.
    """
    msg: dict[str, Any] = {"type": "ack", "action": action, "ok": True}
    if payload:
        msg.update(payload)
    return msg


# ---------------------------------------------------------------------------
# Parsing des messages client
# ---------------------------------------------------------------------------


def parse_client_message(raw: str, config: GameConfig) -> ClientMessage:
    """Parse et valide un message texte brut reçu du client.

    Args:
        raw: Chaîne JSON brute.
        config: Configuration du jeu (pour valider les building_id).

    Returns:
        ClientMessage typé (ActionMsg, ResetMsg, LoadModelMsg, AutoStepMsg).

    Raises:
        ProtocolError: Si le JSON est malformé ou le message invalide.
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"JSON malformé : {exc}") from exc

    if not isinstance(data, dict):
        raise ProtocolError("Le message doit être un objet JSON.")

    msg_type = data.get("type")
    if not isinstance(msg_type, str):
        raise ProtocolError("Champ 'type' manquant ou non-string.")

    if msg_type == "action":
        return _parse_action(data, config)
    if msg_type == "reset":
        return _parse_reset(data)
    if msg_type == "load_model":
        return _parse_load_model(data)
    if msg_type == "auto_step":
        return _parse_auto_step(data)

    raise ProtocolError(f"Type de message inconnu : '{msg_type}'.")


def _parse_action(data: dict[str, Any], config: GameConfig) -> ActionMsg:
    action_data = data.get("action")
    if not isinstance(action_data, dict):
        raise ProtocolError("Champ 'action' manquant ou non-objet.")

    action_type = action_data.get("type")
    if action_type not in ("place", "demolish", "do_nothing"):
        raise ProtocolError(
            f"action.type invalide : '{action_type}'. "
            "Valeurs acceptées : 'place', 'demolish', 'do_nothing'."
        )

    if action_type == "do_nothing":
        return ActionMsg(action=Action("do_nothing"))

    x = action_data.get("x")
    y = action_data.get("y")
    if not isinstance(x, int) or not isinstance(y, int):
        raise ProtocolError("Coordonnées x, y manquantes ou non-entiers.")
    if not (0 <= x < 32 and 0 <= y < 32):
        raise ProtocolError(
            f"Coordonnées hors grille : x={x}, y={y}. Valeurs attendues : 0–31."
        )

    if action_type == "demolish":
        return ActionMsg(action=Action("demolish", x=x, y=y))

    # place
    building_id = action_data.get("building_id")
    if not isinstance(building_id, str):
        raise ProtocolError("Champ 'building_id' manquant ou non-string pour action 'place'.")
    if building_id not in config.buildings.buildings:
        raise ProtocolError(
            f"Bâtiment inconnu : '{building_id}'. "
            f"Bâtiments valides : {sorted(config.buildings.buildings)}."
        )
    return ActionMsg(action=Action("place", building_id=building_id, x=x, y=y))


def _parse_reset(data: dict[str, Any]) -> ResetMsg:
    seed = data.get("seed")
    if seed is not None and not isinstance(seed, int):
        raise ProtocolError("Champ 'seed' doit être un entier ou absent.")
    return ResetMsg(seed=seed)


def _parse_load_model(data: dict[str, Any]) -> LoadModelMsg:
    path = data.get("path")
    if not isinstance(path, str) or not path:
        raise ProtocolError("Champ 'path' manquant ou vide pour 'load_model'.")
    return LoadModelMsg(path=path)


def _parse_auto_step(data: dict[str, Any]) -> AutoStepMsg:
    n = data.get("n", 1)
    if not isinstance(n, int) or n < 1:
        raise ProtocolError("Champ 'n' doit être un entier >= 1.")
    if n > AUTO_STEP_MAX:
        raise ProtocolError(
            f"'n' trop grand : {n} (maximum : {AUTO_STEP_MAX})."
        )
    return AutoStepMsg(n=n)
