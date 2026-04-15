"""Script d'entrainement MaskablePPO pour Vitruvius (run unique Nova Roma)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch.distributions

# sb3-contrib MaskableCategorical produit des probs dont la somme diffère
# légèrement de 1.0 en float32 sur certaines plateformes (Linux/torch>=2.10).
# Désactiver validate_args globalement évite le ValueError sans affecter l'apprentissage.
torch.distributions.Distribution.set_default_validate_args(False)

from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from vitruvius.config import load_config
from vitruvius.rl.gym_env import VitruviusEnv

logger = logging.getLogger(__name__)

DEFAULT_TOTAL_TIMESTEPS: int = 10_000_000
DEFAULT_N_ENVS: int = 8
DEFAULT_MAX_TURNS: int = 1000
DEFAULT_CHECKPOINT_FREQ: int = 100_000
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Callback metriques metier
# ---------------------------------------------------------------------------


class VitruviusMetricsCallback(BaseCallback):
    """Logue des metriques metier Vitruvius dans TensorBoard a chaque fin d'episode.

    Metriques loguees (prefixe vitruvius/) :
        city_level, population, satisfaction, victory_rate,
        defeat_rate, famine_rate, exodus_rate.
    """

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._ep_stats: list[dict] = []

    def _on_step(self) -> bool:
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if not done:
                continue
            result = info.get("turn_result")
            if result is None:
                continue
            self._ep_stats.append({
                "city_level": result.city_level,
                "population": result.total_population,
                "satisfaction": result.global_satisfaction,
                "victory": float(result.victory),
                "defeat": float(result.defeat),
                "famine": float(result.famine_count > 0),
                "exodus": float(result.exodus > 0),
            })
        return True

    def _on_rollout_end(self) -> None:
        if not self._ep_stats:
            return
        for key in self._ep_stats[0]:
            values = [s[key] for s in self._ep_stats]
            self.logger.record(f"vitruvius/{key}", np.mean(values))
        self._ep_stats = []


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def make_env_fn(seed: int, max_turns: int):
    """Retourne une factory sans argument creant un VitruviusEnv.

    Args:
        seed: Seed RNG de base. make_vec_env ajoutera l'indice de l'env.
        max_turns: Nombre de tours avant truncation.

    Returns:
        Callable () -> VitruviusEnv utilisable par make_vec_env.
    """
    config = load_config()

    def _init() -> VitruviusEnv:
        return VitruviusEnv(config=config, seed=seed, max_turns=max_turns)

    return _init


def build_vec_env(
    n_envs: int,
    seed: int,
    max_turns: int,
    subproc: bool,
) -> VecEnv:
    """Cree un VecEnv parallelise avec Monitor wrapper automatique.

    Args:
        n_envs: Nombre d'environnements paralleles.
        seed: Seed de base (chaque env recoit seed + i * 1000 pour eviter
              les collisions lors des retries de generation de terrain).
        max_turns: Nombre de tours avant truncation.
        subproc: Si True, utilise SubprocVecEnv (deconseille sur Windows).

    Returns:
        VecEnv vectorise avec Monitor wrapper.
    """
    vec_cls = SubprocVecEnv if subproc else DummyVecEnv
    config = load_config()
    env_fns = [
        (lambda s=seed + i * 1000: Monitor(VitruviusEnv(config=config, seed=s, max_turns=max_turns)))
        for i in range(n_envs)
    ]
    return vec_cls(env_fns)


def build_model(
    env: VecEnv,
    run_dir: Path,
    args: argparse.Namespace,
) -> MaskablePPO:
    """Instancie MaskablePPO avec hyperparametres configures.

    Args:
        env: VecEnv deja construit.
        run_dir: Dossier racine TensorBoard.
        args: Namespace argparse contenant les hyperparametres.

    Returns:
        Instance MaskablePPO non encore entrainee.
    """
    return MaskablePPO(
        "MultiInputPolicy",
        env,
        learning_rate=lambda f: args.lr_end + (args.learning_rate - args.lr_end) * f,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=str(run_dir),
        verbose=1,
    )


# ---------------------------------------------------------------------------
# Entrainement
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> Path:
    """Lance un entrainement MaskablePPO complet.

    Args:
        args: Namespace argparse issu de build_argparser().

    Returns:
        Chemin du fichier modele final sauvegarde (.zip).
    """
    run_dir = Path(args.run_dir)
    model_dir = Path(args.model_dir) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Demarrage entrainement : %s — %d timesteps, %d envs",
        args.run_name,
        args.total_timesteps,
        args.n_envs,
    )

    env = build_vec_env(args.n_envs, args.seed, args.max_turns, args.subproc)

    if args.resume:
        logger.info("Resume depuis %s", args.resume)
        model = MaskablePPO.load(args.resume, env=env)
    else:
        model = build_model(env, run_dir, args)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(model_dir),
        name_prefix="vitruvius",
        verbose=1,
    )
    metrics_cb = VitruviusMetricsCallback()

    # log_interval : nombre de rollouts entre deux affichages du tableau SB3.
    # Cible : ~32768  steps globaux entre deux logs.
    steps_per_rollout = args.n_steps * args.n_envs
    log_interval = max(1, 32768  // steps_per_rollout)

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=CallbackList([checkpoint_cb, metrics_cb]),
        tb_log_name=args.run_name,
        reset_num_timesteps=not bool(args.resume),
        progress_bar=False,
        log_interval=log_interval,
    )

    final_path = model_dir / "vitruvius_final.zip"
    model.save(str(final_path))
    logger.info("Modele final sauvegarde : %s", final_path)

    env.close()
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    """Construit le parser CLI pour le script d'entrainement.

    Returns:
        ArgumentParser configure avec tous les hyperparametres tunables.
    """
    p = argparse.ArgumentParser(
        description="Entrainement MaskablePPO Vitruvius — run unique Nova Roma",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Entrainement
    p.add_argument(
        "--total-timesteps", type=int, default=DEFAULT_TOTAL_TIMESTEPS,
        help="Nombre total de timesteps d'entrainement",
    )
    p.add_argument(
        "--n-envs", type=int, default=DEFAULT_N_ENVS,
        help="Nombre d'environnements parallelises",
    )
    p.add_argument(
        "--max-turns", type=int, default=DEFAULT_MAX_TURNS,
        help="Nombre de tours avant truncation d'un episode",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Seed RNG de base pour les environnements",
    )
    # Logging / sauvegarde
    p.add_argument("--run-name", type=str, default="nova_roma", help="Nom du run TensorBoard")
    p.add_argument("--run-dir", type=str, default="runs", help="Dossier racine TensorBoard")
    p.add_argument("--model-dir", type=str, default="models", help="Dossier de sauvegarde des modeles")
    p.add_argument(
        "--checkpoint-freq", type=int, default=DEFAULT_CHECKPOINT_FREQ,
        help="Frequence de sauvegarde en timesteps globaux",
    )
    # Parallelisme
    p.add_argument(
        "--subproc", action="store_true",
        help="Utiliser SubprocVecEnv (Linux uniquement — fragile sur Windows)",
    )
    # Resume
    p.add_argument(
        "--resume", type=str, default=None,
        help="Chemin vers un checkpoint .zip pour reprendre l'entrainement",
    )
    # Hyperparametres PPO
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--lr-end", type=float, default=1e-4)
    p.add_argument("--n-steps", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--ent-coef", type=float, default=0.02)
    return p


def main() -> None:
    """Point d'entree CLI : python -m vitruvius.rl.train [options]."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
