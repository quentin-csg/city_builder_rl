"""Serveur WebSocket (port 9876) : expose le game_state JSON à Godot.

Chaque client reçoit sa propre GameSession indépendante.

Protocole :
    Serveur → Client :
        init   — à la connexion et après reset (seed, terrain, catalog, state)
        state  — après chaque action (state + turn_result)
        ack    — après load_model
        error  — message invalide (connexion maintenue)

    Client → Serveur :
        action     — place / demolish / do_nothing
        reset      — nouvelle partie
        load_model — charger un modèle MaskablePPO (.zip)
        auto_step  — avancer de n tours avec le modèle chargé

Usage :
    python -m vitruvius.bridge.server [--host HOST] [--port PORT] [--seed SEED]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import websockets

from vitruvius.bridge.protocol import (
    ActionMsg,
    AutoStepMsg,
    ClientMessage,
    LoadModelMsg,
    ProtocolError,
    ResetMsg,
    build_ack_message,
    build_error_message,
    build_init_message,
    build_state_message,
    parse_client_message,
)
from vitruvius.config import load_config
from vitruvius.engine.actions import decode_action, get_building_order
from vitruvius.engine.game_state import GameState, init_game_state
from vitruvius.engine.turn import Action, TurnResult, step

if TYPE_CHECKING:
    from vitruvius.config import GameConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GameSession — logique d'une partie
# ---------------------------------------------------------------------------


class GameSession:
    """État d'une partie associée à une connexion client.

    Attributes:
        gs: État courant du jeu.
        config: Configuration unifiée (bâtiments, ressources, etc.).
        building_list: Liste ordonnée des bâtiments (pour decode_action).
        building_index_map: building_id → index (pour build_observation).
        model: Modèle MaskablePPO chargé, ou None en mode humain.
    """

    def __init__(self, config: GameConfig, seed: int) -> None:
        self.config = config
        self.building_list, self.building_index_map = get_building_order(config)
        self.model: Any = None  # MaskablePPO | None
        self.gs: GameState = init_game_state(config, seed=seed)
        self._prev_pop: int = 0

    # ------------------------------------------------------------------
    # Actions humaines
    # ------------------------------------------------------------------

    def apply_action(self, action: Action) -> TurnResult:
        """Applique une action et retourne le résultat du tour.

        Args:
            action: Action choisie par le client.

        Returns:
            TurnResult décrivant les effets du tour.
        """
        return step(self.gs, self.config, action)

    def reset(self, seed: int | None) -> None:
        """Réinitialise la partie.

        Args:
            seed: Nouveau seed. Si None, réutilise le seed actuel.
        """
        new_seed = seed if seed is not None else self.gs.seed
        self.gs = init_game_state(self.config, seed=new_seed)
        self.model = None
        self._prev_pop = 0

    # ------------------------------------------------------------------
    # Replay d'un modèle
    # ------------------------------------------------------------------

    def load_model(self, path: str) -> None:
        """Charge un modèle MaskablePPO depuis un fichier .zip.

        Import lazy de sb3_contrib pour ne pas pénaliser le démarrage
        du serveur quand le mode replay n'est pas utilisé.

        Args:
            path: Chemin vers le fichier .zip (relatif au cwd).

        Raises:
            FileNotFoundError: Si le fichier n'existe pas.
            ValueError: Si le fichier ne peut pas être chargé.
        """
        from sb3_contrib import MaskablePPO  # lazy import

        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle introuvable : {model_path.resolve()}")
        self.model = MaskablePPO.load(str(model_path))
        logger.info("Modèle chargé : %s", model_path)

    def auto_step(self, n: int = 1) -> list[TurnResult]:
        """Avance de n tours avec le modèle chargé.

        Boucle : build_observation → action_mask → model.predict → decode_action → step.
        S'arrête si gs.done devient True avant n tours.

        Args:
            n: Nombre de tours à simuler (1 ≤ n ≤ AUTO_STEP_MAX).

        Returns:
            Liste des TurnResult produits (len ≤ n).

        Raises:
            RuntimeError: Si aucun modèle n'est chargé.
        """
        if self.model is None:
            raise RuntimeError("Aucun modèle chargé. Envoyez 'load_model' d'abord.")

        from vitruvius.engine.actions import compute_action_mask
        from vitruvius.rl.observation import build_observation

        results: list[TurnResult] = []
        dynamics: dict[str, float] = {
            "growth_rate": 0.0,
            "wheat_conso_ratio": 0.0,
            "net_income": 0.0,
        }

        for _ in range(n):
            if self.gs.done:
                break

            obs = build_observation(
                self.gs, self.config, self.building_index_map, dynamics
            )
            mask = compute_action_mask(self.gs, self.config, self.building_list)

            action_int, _ = self.model.predict(obs, action_masks=mask, deterministic=True)
            action = decode_action(int(action_int), self.building_list)
            result = step(self.gs, self.config, action)
            results.append(result)

            # Mise à jour des dynamics pour le tour suivant — formules alignées sur gym_env.py
            new_pop = result.total_population
            growth_rate = (new_pop - self._prev_pop) / max(1, self._prev_pop)
            growth_rate = max(-1.0, min(1.0, growth_rate))
            self._prev_pop = new_pop

            wheat_conso = sum(
                math.ceil(h.population / 10) for h in self.gs.houses.values()
            )
            wheat_stock_before = self.gs.resource_state.wheat + wheat_conso
            conso_ratio = wheat_conso / max(1, wheat_stock_before)
            conso_ratio = max(0.0, min(2.0, conso_ratio)) / 2.0

            net_income = max(
                -1.0,
                min(1.0, (result.taxes_collected - result.maintenance_paid) / 1000.0),
            )

            dynamics = {
                "growth_rate": growth_rate,
                "wheat_conso_ratio": conso_ratio,
                "net_income": net_income,
            }

        return results


# ---------------------------------------------------------------------------
# Gestionnaire WebSocket par client
# ---------------------------------------------------------------------------


async def _send(ws: websockets.asyncio.server.ServerConnection, msg: dict[str, Any]) -> None:
    """Sérialise et envoie un message JSON au client."""
    await ws.send(json.dumps(msg))


async def handle_client(
    ws: websockets.asyncio.server.ServerConnection, config: GameConfig, initial_seed: int
) -> None:
    """Gère la connexion d'un client WebSocket.

    Crée une GameSession dédiée, envoie l'init, puis boucle sur les messages
    entrants jusqu'à déconnexion.

    Args:
        ws: WebSocket du client.
        config: Configuration du jeu.
        initial_seed: Seed pour la première partie.
    """
    remote = ws.remote_address
    logger.info("Client connecté : %s", remote)

    session = GameSession(config, seed=initial_seed)
    await _send(ws, build_init_message(session.gs, config, model_loaded=False))

    try:
        async for raw in ws:
            try:
                msg: ClientMessage = parse_client_message(raw, config)
            except ProtocolError as exc:
                await _send(ws, build_error_message(str(exc)))
                continue

            # Dispatch
            if isinstance(msg, ActionMsg):
                try:
                    result = session.apply_action(msg.action)
                    await _send(ws, build_state_message(session.gs, result))
                except Exception as exc:
                    logger.exception("Erreur apply_action")
                    await _send(ws, build_error_message(f"Erreur interne : {exc}"))

            elif isinstance(msg, ResetMsg):
                session.reset(msg.seed)
                await _send(ws, build_init_message(session.gs, config, model_loaded=False))

            elif isinstance(msg, LoadModelMsg):
                try:
                    session.load_model(msg.path)
                    await _send(ws, build_ack_message("load_model", {"path": msg.path}))
                except FileNotFoundError as exc:
                    await _send(ws, build_error_message(str(exc)))
                except Exception as exc:
                    logger.exception("Erreur load_model")
                    await _send(ws, build_error_message(f"Impossible de charger le modèle : {exc}"))

            elif isinstance(msg, AutoStepMsg):
                try:
                    results = session.auto_step(msg.n)
                    # Envoie un message state pour chaque tour — Godot peut animer
                    for result in results:
                        await _send(ws, build_state_message(session.gs, result))
                        if session.gs.done:
                            break
                except RuntimeError as exc:
                    await _send(ws, build_error_message(str(exc)))
                except Exception as exc:
                    logger.exception("Erreur auto_step")
                    await _send(ws, build_error_message(f"Erreur auto_step : {exc}"))

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client déconnecté : %s", remote)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


async def serve(host: str = "localhost", port: int = 9876, seed: int = 42) -> None:
    """Démarre le serveur WebSocket.

    Args:
        host: Adresse d'écoute (défaut : localhost — dev uniquement, pas de TLS).
        port: Port TCP (défaut : 9876).
        seed: Seed initial pour les nouvelles sessions.
    """
    config = load_config()

    async def _handler(ws: websockets.asyncio.server.ServerConnection) -> None:
        await handle_client(ws, config, initial_seed=seed)

    logger.info("Serveur WebSocket démarré sur ws://%s:%d", host, port)
    async with websockets.serve(_handler, host, port):
        await asyncio.Future()  # tourne indéfiniment


def main() -> None:
    """Point d'entrée CLI.

    Usage :
        python -m vitruvius.bridge.server [--host HOST] [--port PORT] [--seed SEED]
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Serveur WebSocket Nova Roma → Godot")
    parser.add_argument("--host", default="localhost", help="Adresse d'écoute (défaut : localhost)")
    parser.add_argument("--port", type=int, default=9876, help="Port TCP (défaut : 9876)")
    parser.add_argument("--seed", type=int, default=42, help="Seed de la partie (défaut : 42)")
    args = parser.parse_args()

    try:
        asyncio.run(serve(host=args.host, port=args.port, seed=args.seed))
    except KeyboardInterrupt:
        logger.info("Serveur arrêté.")


if __name__ == "__main__":
    main()
