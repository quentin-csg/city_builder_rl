"""Script d'evaluation MaskablePPO pour Vitruvius."""

from __future__ import annotations

import argparse
import logging
import statistics
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO

from vitruvius.config import load_config
from vitruvius.rl.gym_env import VitruviusEnv

logger = logging.getLogger(__name__)

DEFAULT_N_EPISODES: int = 20
DEFAULT_MAX_TURNS: int = 1000
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------


def run_episode(
    model: MaskablePPO,
    env: VitruviusEnv,
    seed: int,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Joue un episode complet et retourne ses statistiques.

    Args:
        model: MaskablePPO deja charge.
        env: VitruviusEnv (sera reset avec seed).
        seed: Seed pour le reset de l'episode.
        deterministic: Si True, politique argmax ; sinon echantillonnage stochastique.

    Returns:
        Dict de stats : total_reward, episode_length, victory, defeat,
        bankrupt, final_city_level, final_population, final_satisfaction,
        final_denarii, final_turn.
    """
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    last_result = None

    terminated = False
    truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        action, _ = model.predict(obs, action_masks=mask, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        steps += 1
        last_result = info["turn_result"]

    gs = env.gs
    return {
        "total_reward": total_reward,
        "episode_length": steps,
        "victory": bool(last_result.victory) if last_result else False,
        "defeat": bool(last_result.defeat) if last_result else False,
        "bankrupt": bool(last_result.bankrupt) if last_result else False,
        "final_city_level": gs.city_level,
        "final_population": sum(h.population for h in gs.houses.values()),
        "final_satisfaction": gs.global_satisfaction,
        "final_denarii": gs.resource_state.denarii,
        "final_turn": gs.turn,
    }


def aggregate_stats(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrege une liste de stats d'episodes en metriques globales.

    Args:
        episodes: Liste de dicts retournes par run_episode.

    Returns:
        Dict avec moyennes/medianes/std pour les metriques numeriques,
        taux pour les booleens (victory_rate, defeat_rate, bankrupt_rate).
    """
    n = len(episodes)
    rewards = [e["total_reward"] for e in episodes]
    lengths = [e["episode_length"] for e in episodes]
    levels = [e["final_city_level"] for e in episodes]
    pops = [e["final_population"] for e in episodes]
    sats = [e["final_satisfaction"] for e in episodes]

    return {
        "n_episodes": n,
        "victory_rate": sum(1 for e in episodes if e["victory"]) / n,
        "defeat_rate": sum(1 for e in episodes if e["defeat"]) / n,
        "bankrupt_rate": sum(1 for e in episodes if e["bankrupt"]) / n,
        "reward_mean": statistics.mean(rewards),
        "reward_median": statistics.median(rewards),
        "reward_std": statistics.stdev(rewards) if n > 1 else 0.0,
        "reward_min": min(rewards),
        "reward_max": max(rewards),
        "length_mean": statistics.mean(lengths),
        "length_median": statistics.median(lengths),
        "city_level_mean": statistics.mean(levels),
        "city_level_max": max(levels),
        "population_mean": statistics.mean(pops),
        "population_max": max(pops),
        "satisfaction_mean": statistics.mean(sats),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Charge un modele et le fait jouer N episodes.

    Args:
        args: Namespace argparse issu de build_argparser().

    Returns:
        Dict de stats agregees (voir aggregate_stats).

    Raises:
        FileNotFoundError: Si le fichier modele n'existe pas.
    """
    model_path = Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Modele introuvable : {model_path}")

    logger.info("Chargement du modele : %s", model_path)
    model = MaskablePPO.load(str(model_path))

    config = load_config()
    episodes: list[dict[str, Any]] = []

    for i in range(args.n_episodes):
        seed = args.seed + i * 1000
        env = VitruviusEnv(config=config, seed=seed, max_turns=args.max_turns)
        stats = run_episode(model, env, seed=seed, deterministic=not args.stochastic)
        env.close()
        episodes.append(stats)
        logger.info(
            "Episode %d/%d (seed %d) : reward=%.2f len=%d level=%d pop=%d victory=%s",
            i + 1, args.n_episodes, seed,
            stats["total_reward"], stats["episode_length"],
            stats["final_city_level"], stats["final_population"],
            stats["victory"],
        )

    agg = aggregate_stats(episodes)
    logger.info("=== Stats agregees sur %d episodes ===", agg["n_episodes"])
    logger.info(
        "Reward    : mean=%.2f  median=%.2f  std=%.2f  min=%.2f  max=%.2f",
        agg["reward_mean"], agg["reward_median"], agg["reward_std"],
        agg["reward_min"], agg["reward_max"],
    )
    logger.info(
        "Longueur  : mean=%.1f  median=%.1f",
        agg["length_mean"], agg["length_median"],
    )
    logger.info(
        "Ville     : level_mean=%.2f  level_max=%d",
        agg["city_level_mean"], agg["city_level_max"],
    )
    logger.info(
        "Population: mean=%.0f  max=%d",
        agg["population_mean"], agg["population_max"],
    )
    logger.info("Satisfaction moyenne : %.2f", agg["satisfaction_mean"])
    logger.info(
        "Taux      : victory=%.0f%%  defeat=%.0f%%  bankrupt=%.0f%%",
        agg["victory_rate"] * 100, agg["defeat_rate"] * 100,
        agg["bankrupt_rate"] * 100,
    )
    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour le script d'evaluation.

    Returns:
        ArgumentParser configure.
    """
    p = argparse.ArgumentParser(
        description="Evaluation d'un modele MaskablePPO Vitruvius",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("model_path", type=str, help="Chemin vers le .zip du modele")
    p.add_argument(
        "--n-episodes", type=int, default=DEFAULT_N_EPISODES,
        help="Nombre d'episodes a jouer",
    )
    p.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="Nombre de tours avant truncation",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed de base (chaque episode recoit seed + i*1000)",
    )
    p.add_argument(
        "--stochastic", action="store_true",
        help="Politique stochastique au lieu de deterministe",
    )
    return p


def main() -> None:
    """Point d'entree CLI : python -m vitruvius.rl.evaluate <model_path> [options]."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_argparser().parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
