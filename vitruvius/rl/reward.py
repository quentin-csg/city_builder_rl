"""Reward shaping : récompenses intermédiaires et terminales pour MaskablePPO."""

from __future__ import annotations

from dataclasses import dataclass

from vitruvius.engine.turn import TurnResult


@dataclass
class RewardState:
    """Snapshot des métriques nécessaires au calcul des deltas inter-tour."""

    total_population: int
    city_level: int
    global_satisfaction: float
    housing_sum: int  # somme des niveaux de toutes les maisons


# ---------------------------------------------------------------------------
# Coefficients (voir CLAUDE.md — reward shaping)
# ---------------------------------------------------------------------------

W_POP = 1.0
W_LEVEL = 5.0
W_SAT = 0.5
W_HOUSING = 0.1
W_BANKRUPT = -0.5
W_FAMINE = -0.3
W_EXODUS = -0.2
W_VICTORY = 100.0
W_DEFEAT = -10.0
W_SURVIVAL = 0.01


def compute_reward(
    prev: RewardState,
    curr: RewardState,
    result: TurnResult,
) -> float:
    """Calcule le reward d'un tour selon la formule CLAUDE.md.

    Fonction pure : aucun effet de bord, aucune dépendance à GameState.

    Args:
        prev: Snapshot RewardState avant l'action (capturé juste avant step).
        curr: Snapshot RewardState après l'action (capturé juste après step).
        result: TurnResult retourné par engine step, portant les flags du tour.

    Returns:
        Reward scalaire (non clampé).
    """
    delta_pop = curr.total_population - prev.total_population
    delta_level = curr.city_level - prev.city_level
    delta_sat = curr.global_satisfaction - prev.global_satisfaction
    delta_housing = curr.housing_sum - prev.housing_sum

    reward = 0.0
    reward += W_POP * (delta_pop / 100.0)
    reward += W_LEVEL * delta_level
    reward += W_SAT * delta_sat
    reward += W_HOUSING * (delta_housing / 10.0)

    if result.bankrupt:
        reward += W_BANKRUPT
    if result.famine_count > 0:
        reward += W_FAMINE
    if result.exodus > 0:
        reward += W_EXODUS

    if result.victory:
        reward += W_VICTORY
    if result.defeat:
        reward += W_DEFEAT

    reward += W_SURVIVAL
    return float(reward)
