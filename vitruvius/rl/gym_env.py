"""Wrapper Gymnasium standard : reset, step, observation space, action space."""

from __future__ import annotations

import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from vitruvius.config import load_config
from vitruvius.engine.actions import (
    TOTAL_ACTIONS,
    compute_action_mask,
    decode_action,
    get_building_order,
)
from vitruvius.engine.game_state import GameState, init_game_state
from vitruvius.engine.turn import step as engine_step
from vitruvius.rl.observation import build_observation
from vitruvius.rl.reward import RewardState, compute_reward

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vitruvius.config import GameConfig


class VitruviusEnv(gym.Env):
    """Environnement Gymnasium pour le city builder Nova Roma.

    Compatible MaskablePPO (sb3-contrib) via `action_masks()`.

    Args:
        config: Configuration du jeu. Si None, charge via load_config().
        seed: Seed RNG pour la génération du terrain. None = aléatoire.
        max_turns: Nombre de tours avant truncation (défaut 1000).
    """

    metadata: dict = {"render_modes": []}

    def __init__(
        self,
        config: GameConfig | None = None,
        seed: int | None = None,
        max_turns: int = 1000,
    ) -> None:
        super().__init__()
        self.config = config or load_config()
        self._seed = seed
        self.max_turns = max_turns
        self.building_list, self.building_index_map = get_building_order(self.config)

        self.observation_space = spaces.Dict({
            "grid": spaces.Box(
                low=0.0, high=1.0, shape=(32, 32, 31), dtype=np.float32
            ),
            "global_features": spaces.Box(
                low=-1.0, high=1.0, shape=(18,), dtype=np.float32
            ),
        })
        self.action_space = spaces.Discrete(TOTAL_ACTIONS)

        self.gs: GameState | None = None
        self._prev_pop: int = 0
        self._last_dynamics: dict[str, float] = {
            "growth_rate": 0.0,
            "wheat_conso_ratio": 0.0,
            "net_income": 0.0,
        }
        self._prev_reward_state: RewardState | None = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict, dict]:
        """Réinitialise l'environnement.

        Args:
            seed: Seed pour ce reset (prioritaire sur self._seed).
            options: Ignoré (futur usage).

        Returns:
            Tuple (observation, info).
        """
        super().reset(seed=seed)
        actual_seed = seed if seed is not None else self._seed
        self.gs = init_game_state(self.config, seed=actual_seed)
        self._prev_pop = 0
        self._last_dynamics = {
            "growth_rate": 0.0,
            "wheat_conso_ratio": 0.0,
            "net_income": 0.0,
        }
        self._prev_reward_state = self._snapshot_reward_state()
        obs = build_observation(
            self.gs, self.config, self.building_index_map, self._last_dynamics
        )
        return obs, {}

    def step(self, action_int: int) -> tuple[dict, float, bool, bool, dict]:
        """Exécute une action et avance d'un tour.

        Args:
            action_int: Entier dans [0, TOTAL_ACTIONS).

        Returns:
            Tuple (obs, reward, terminated, truncated, info).
        """
        if self.gs is None:
            raise RuntimeError("reset() must be called before step().")
        action = decode_action(int(action_int), self.building_list)
        result = engine_step(self.gs, self.config, action)

        # Calcul des dynamics inter-tour
        new_pop = result.total_population
        growth_rate = (new_pop - self._prev_pop) / max(1, self._prev_pop)
        growth_rate = float(np.clip(growth_rate, -1.0, 1.0))

        wheat_conso = sum(
            math.ceil(h.population / 10) for h in self.gs.houses.values()
        )
        wheat_stock_before = self.gs.resource_state.wheat + wheat_conso
        conso_ratio = wheat_conso / max(1, wheat_stock_before)
        conso_ratio = float(np.clip(conso_ratio, 0.0, 2.0)) / 2.0

        net_income = float(
            np.clip(
                (result.taxes_collected - result.maintenance_paid) / 1_000.0,
                -1.0,
                1.0,
            )
        )

        self._last_dynamics = {
            "growth_rate": growth_rate,
            "wheat_conso_ratio": conso_ratio,
            "net_income": net_income,
        }
        self._prev_pop = new_pop

        obs = build_observation(
            self.gs, self.config, self.building_index_map, self._last_dynamics
        )
        curr_state = self._snapshot_reward_state()
        reward = compute_reward(self._prev_reward_state, curr_state, result)
        self._prev_reward_state = curr_state
        terminated = bool(result.done)
        truncated = bool(self.gs.turn >= self.max_turns)

        info: dict = {
            "turn_result": result,
            "action_mask": self.action_masks().copy(),
        }
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Retourne le masque d'actions valides pour MaskablePPO.

        Returns:
            Array booléen de forme (TOTAL_ACTIONS,).

        Raises:
            RuntimeError: Si reset() n'a pas encore été appelé.
        """
        if self.gs is None:
            raise RuntimeError("reset() must be called before action_masks().")
        return compute_action_mask(self.gs, self.config, self.building_list)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_reward_state(self) -> RewardState:
        """Capture un snapshot des métriques pour le calcul du reward delta.

        Les flags one-shot (milestones) sont irreversibles : une fois True,
        ils restent True même si le batiment est démoli.
        """
        ids = self.gs.grid._placed_ids  # Counter[str], O(1) par type
        prev = self._prev_reward_state

        # Propagation irreversible : prev.flag OR flag_actuel
        def _keep(prev_val: bool, current: bool) -> bool:
            return prev_val or current

        p_has_forum = prev.has_forum if prev else False
        p_has_obelisk = prev.has_obelisk if prev else False
        p_has_prefecture = prev.has_prefecture if prev else False
        p_house = prev.first_house_placed if prev else False
        p_farm = prev.first_farm_placed if prev else False
        p_well = prev.first_well_placed if prev else False
        p_temple = prev.first_temple_placed if prev else False
        p_granary = prev.first_granary_placed if prev else False
        p_market = prev.first_market_placed if prev else False
        p_lumber = prev.first_lumber_camp_placed if prev else False
        p_trading = prev.first_trading_post_placed if prev else False
        p_population = prev.first_population if prev else False
        p_marble_quarry = prev.first_marble_quarry_placed if prev else False
        p_warehouse_marble = prev.first_warehouse_marble_placed if prev else False
        p_baths = prev.first_baths_placed if prev else False
        p_theater = prev.first_theater_placed if prev else False

        total_pop = sum(h.population for h in self.gs.houses.values())

        return RewardState(
            total_population=total_pop,
            city_level=self.gs.city_level,
            global_satisfaction=self.gs.global_satisfaction,
            housing_sum=sum(h.level for h in self.gs.houses.values()),
            has_forum=_keep(p_has_forum, ids["forum"] > 0),
            has_obelisk=_keep(p_has_obelisk, ids["obelisk"] > 0),
            has_prefecture=_keep(p_has_prefecture, ids["prefecture"] > 0),
            first_house_placed=_keep(p_house, len(self.gs.houses) > 0),
            first_farm_placed=_keep(p_farm, ids["wheat_farm"] > 0),
            first_well_placed=_keep(p_well, ids["well"] > 0),
            first_temple_placed=_keep(p_temple, ids["temple"] > 0),
            first_granary_placed=_keep(p_granary, ids["granary"] > 0),
            first_market_placed=_keep(p_market, ids["market"] > 0),
            first_lumber_camp_placed=_keep(p_lumber, ids["lumber_camp"] > 0),
            first_trading_post_placed=_keep(p_trading, ids["trading_post"] > 0),
            first_population=_keep(p_population, total_pop > 0),
            first_marble_quarry_placed=_keep(p_marble_quarry, ids["marble_quarry"] > 0),
            first_warehouse_marble_placed=_keep(p_warehouse_marble, ids["warehouse_marble"] > 0),
            first_baths_placed=_keep(p_baths, ids["baths"] > 0),
            first_theater_placed=_keep(p_theater, ids["theater"] > 0),
            marble_stock=self.gs.resource_state.marble,
        )
