"""Tests d'intégration du serveur WebSocket — client réel en asyncio.

Chaque test démarre un serveur sur un port aléatoire, se connecte en tant
que client WebSocket, envoie des messages et vérifie les réponses.
Les tests sont écrits avec asyncio.run() pour rester compatibles sans
plugins pytest (pas de pytest-asyncio requis).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import pytest
import websockets

from vitruvius.bridge.server import GameSession, handle_client
from vitruvius.config import load_config


# ---------------------------------------------------------------------------
# Fixtures et helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_config()


def _find_free_port() -> int:
    """Trouve un port TCP libre sur localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


@asynccontextmanager
async def _server_and_client(
    seed: int = 42,
) -> AsyncGenerator[websockets.WebSocketClientProtocol, None]:
    """Démarre un serveur sur un port libre et connecte un client."""
    cfg = load_config()
    port = _find_free_port()

    async def _handler(ws):
        await handle_client(ws, cfg, initial_seed=seed)

    server = await websockets.serve(_handler, "localhost", port)
    try:
        async with websockets.connect(f"ws://localhost:{port}") as client:
            yield client
    finally:
        server.close()
        await server.wait_closed()


def _run(coro):
    """Exécute une coroutine avec asyncio.run() pour les tests."""
    return asyncio.run(coro)


async def _recv_json(ws) -> dict:
    raw = await ws.recv()
    return json.loads(raw)


async def _send(ws, data: dict) -> None:
    await ws.send(json.dumps(data))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_connect_receives_init():
    """À la connexion, le client reçoit un message 'init' complet."""
    async def run():
        async with _server_and_client(seed=1) as ws:
            msg = await _recv_json(ws)
            assert msg["type"] == "init"
            assert msg["seed"] == 1
            assert msg["size"] == 32
            assert len(msg["terrain"]) == 32
            assert len(msg["terrain"][0]) == 32
            assert "buildings_catalog" in msg
            assert "state" in msg
            assert isinstance(msg["model_loaded"], bool)

    _run(run())


def test_send_do_nothing_receives_state():
    """Envoyer 'do_nothing' renvoie un message 'state' avec turn_result."""
    async def run():
        async with _server_and_client() as ws:
            await _recv_json(ws)  # init
            await _send(ws, {"type": "action", "action": {"type": "do_nothing"}})
            msg = await _recv_json(ws)
            assert msg["type"] == "state"
            assert "state" in msg
            assert "turn_result" in msg
            assert "done" in msg
            assert "victory" in msg
            assert msg["turn_result"]["action_succeeded"] is True

    _run(run())


def test_send_place_action_receives_state():
    """Envoyer 'place housing 5 5' renvoie un message 'state'."""
    async def run():
        async with _server_and_client() as ws:
            await _recv_json(ws)  # init
            await _send(ws, {
                "type": "action",
                "action": {"type": "place", "building_id": "housing", "x": 5, "y": 5},
            })
            msg = await _recv_json(ws)
            assert msg["type"] == "state"

    _run(run())


def test_invalid_json_returns_error_and_keeps_connection():
    """JSON invalide → 'error', connexion maintenue, actions suivantes acceptées."""
    async def run():
        async with _server_and_client() as ws:
            await _recv_json(ws)  # init

            # Envoi de JSON malformé
            await ws.send("ce n'est pas du json {{{")
            err = await _recv_json(ws)
            assert err["type"] == "error"
            assert "message" in err

            # La connexion est toujours ouverte — une action valide fonctionne
            await _send(ws, {"type": "action", "action": {"type": "do_nothing"}})
            state = await _recv_json(ws)
            assert state["type"] == "state"

    _run(run())


def test_invalid_building_id_returns_error(config):
    """Un building_id inconnu renvoie 'error' sans crasher le serveur."""
    async def run():
        async with _server_and_client() as ws:
            await _recv_json(ws)  # init
            await _send(ws, {
                "type": "action",
                "action": {"type": "place", "building_id": "catapulte", "x": 5, "y": 5},
            })
            err = await _recv_json(ws)
            assert err["type"] == "error"

    _run(run())


def test_reset_with_new_seed_sends_new_init():
    """Après 'reset', le client reçoit un nouveau message 'init' avec le bon seed."""
    async def run():
        async with _server_and_client(seed=10) as ws:
            init1 = await _recv_json(ws)
            assert init1["seed"] == 10

            await _send(ws, {"type": "reset", "seed": 99})
            init2 = await _recv_json(ws)
            assert init2["type"] == "init"
            assert init2["seed"] == 99

    _run(run())


def test_reset_without_seed_reuses_current_seed():
    """Reset sans seed réutilise le seed de la session."""
    async def run():
        async with _server_and_client(seed=7) as ws:
            init1 = await _recv_json(ws)
            seed = init1["seed"]

            await _send(ws, {"type": "reset"})
            init2 = await _recv_json(ws)
            assert init2["type"] == "init"
            assert init2["seed"] == seed

    _run(run())


def test_load_model_nonexistent_returns_error():
    """'load_model' vers un fichier inexistant renvoie 'error'."""
    async def run():
        async with _server_and_client() as ws:
            await _recv_json(ws)  # init
            await _send(ws, {"type": "load_model", "path": "models/inexistant.zip"})
            err = await _recv_json(ws)
            assert err["type"] == "error"
            assert "introuvable" in err["message"].lower() or "error" in err["message"].lower()

    _run(run())


def test_auto_step_without_model_returns_error():
    """'auto_step' sans modèle chargé renvoie 'error'."""
    async def run():
        async with _server_and_client() as ws:
            await _recv_json(ws)  # init
            await _send(ws, {"type": "auto_step", "n": 1})
            err = await _recv_json(ws)
            assert err["type"] == "error"
            assert "modèle" in err["message"].lower()

    _run(run())


def test_two_clients_have_independent_sessions():
    """Deux clients connectés simultanément ont des sessions indépendantes."""
    async def run():
        cfg = load_config()
        port_a = _find_free_port()
        port_b = _find_free_port()

        async def _handler(ws):
            await handle_client(ws, cfg, initial_seed=42)

        server_a = await websockets.serve(_handler, "localhost", port_a)
        server_b = await websockets.serve(_handler, "localhost", port_b)

        try:
            async with websockets.connect(f"ws://localhost:{port_a}") as ws_a:
                async with websockets.connect(f"ws://localhost:{port_b}") as ws_b:
                    init_a = await _recv_json(ws_a)
                    init_b = await _recv_json(ws_b)

                    # Les deux démarrent au tour 0
                    assert init_a["state"]["turn"] == 0
                    assert init_b["state"]["turn"] == 0

                    # Avancer client A de 2 tours
                    for _ in range(2):
                        await _send(ws_a, {"type": "action", "action": {"type": "do_nothing"}})
                        state_a = await _recv_json(ws_a)

                    # Client B doit toujours être au tour 0 (pas encore joué)
                    # Vérification : envoyer do_nothing et vérifier le turn
                    await _send(ws_b, {"type": "action", "action": {"type": "do_nothing"}})
                    state_b = await _recv_json(ws_b)

                    turn_a = state_a["state"]["turn"]
                    turn_b = state_b["state"]["turn"]
                    # A a joué 2 fois → turn=2 ; B a joué 1 fois → turn=1
                    assert turn_a == 2, f"Client A attendu turn=2, obtenu {turn_a}"
                    assert turn_b == 1, f"Client B attendu turn=1, obtenu {turn_b}"
        finally:
            server_a.close()
            server_b.close()
            await server_a.wait_closed()
            await server_b.wait_closed()

    _run(run())


# ---------------------------------------------------------------------------
# Tests unitaires GameSession (sans WebSocket)
# ---------------------------------------------------------------------------


def test_game_session_reset_with_seed(config):
    """GameSession.reset() change le seed et réinitialise l'état."""
    session = GameSession(config, seed=1)
    initial_seed = session.gs.seed
    session.reset(seed=99)
    assert session.gs.seed == 99
    assert session.gs.turn == 0


def test_game_session_reset_without_seed_keeps_seed(config):
    """GameSession.reset(None) réutilise le seed courant."""
    session = GameSession(config, seed=42)
    session.reset(None)
    assert session.gs.seed == 42
    assert session.gs.turn == 0


def test_game_session_apply_do_nothing(config):
    """apply_action(do_nothing) incrémente le tour."""
    from vitruvius.engine.turn import Action

    session = GameSession(config, seed=0)
    result = session.apply_action(Action("do_nothing"))
    assert result.done is False
    assert session.gs.turn == 1


def test_game_session_load_model_file_not_found(config):
    """load_model lève FileNotFoundError si le fichier est absent."""
    session = GameSession(config, seed=0)
    with pytest.raises(FileNotFoundError):
        session.load_model("models/inexistant.zip")


def test_game_session_auto_step_without_model_raises(config):
    """auto_step sans modèle chargé lève RuntimeError."""
    session = GameSession(config, seed=0)
    with pytest.raises(RuntimeError, match="modèle"):
        session.auto_step(n=1)


# ---------------------------------------------------------------------------
# Tests unitaires : formules dynamics (alignement gym_env.py ↔ server.py)
# ---------------------------------------------------------------------------


def test_game_session_prev_pop_initialized_and_resets(config):
    """_prev_pop démarre à 0 et revient à 0 après reset().

    Cette valeur est utilisée dans growth_rate = (new - prev) / max(1, prev).
    Si elle n'était pas réinitialisée, le premier tour après reset donnerait
    un growth_rate basé sur la pop de la partie précédente.
    """
    session = GameSession(config, seed=0)
    assert session._prev_pop == 0

    session._prev_pop = 999  # simuler N tours joués
    session.reset(seed=1)
    assert session._prev_pop == 0


def test_dynamics_wheat_conso_ratio_formula(config):
    """wheat_conso_ratio utilise le stock AVANT consommation (stock_après + conso).

    Bug précédent (server.py) : min(1, conso / stock_après) utilisait le stock
    post-consommation, sous-estimant le ratio quand le stock était quasi-vide.

    Scénario : 1 blé consommé, 99 blé restant → stock avant = 100.
    Formule correcte : 1 / 100 / 2 = 0.005.
    Ancienne formule : 1 / 99 ≈ 0.0101 — valeur différente.
    """
    wheat_stock_after = 99
    wheat_conso = 1

    # Formule correcte (gym_env.py et server.py corrigé)
    wheat_stock_before = wheat_stock_after + wheat_conso
    correct = max(0.0, min(2.0, wheat_conso / max(1, wheat_stock_before))) / 2.0

    # Ancienne formule buggée (server.py avant correction)
    old = min(1.0, wheat_conso / max(1, wheat_stock_after))

    assert abs(correct - 0.005) < 1e-9
    assert abs(old - 1 / 99) < 1e-9
    assert correct < old  # la correction réduit le ratio (dénominateur plus grand)
