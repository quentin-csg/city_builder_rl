"""Reward shaping : récompenses intermédiaires et terminales pour MaskablePPO."""

from __future__ import annotations

from dataclasses import dataclass, field

from vitruvius.engine.turn import TurnResult


@dataclass
class RewardState:
    """Snapshot des métriques nécessaires au calcul des deltas inter-tour."""

    total_population: int
    city_level: int
    global_satisfaction: float
    housing_sum: int  # somme des niveaux de toutes les maisons

    # Flags de batiments-cles pour la victoire (one-shot, irreversibles)
    has_forum: bool = False
    has_obelisk: bool = False
    has_prefecture: bool = False

    # Milestones de premier placement (one-shot, irreversibles)
    first_house_placed: bool = False
    first_farm_placed: bool = False
    first_well_placed: bool = False
    first_temple_placed: bool = False
    first_granary_placed: bool = False
    first_market_placed: bool = False
    first_lumber_camp_placed: bool = False
    first_trading_post_placed: bool = False

    # Milestone première population
    first_population: bool = False

    # Milestones chaîne marble (level 2 et au-delà)
    first_marble_quarry_placed: bool = False
    first_warehouse_marble_placed: bool = False

    # Milestones services level 4 (baths=hygiene, theater=entertainment)
    first_baths_placed: bool = False
    first_theater_placed: bool = False

    # Stock de marbre courant (pour reward continu marble_progress)
    marble_stock: int = 0


# ---------------------------------------------------------------------------
# Coefficients
# ---------------------------------------------------------------------------

W_POP: float = 8.0        # par tranche de 100 hab gagnés
W_LEVEL: float = 15.0    # par niveau de ville gagné
W_SAT: float = 2.0        # par point de satisfaction gagné (0–1)
W_HOUSING: float = 1.0    # par tranche de 10 niveaux de maison gagnés
W_BANKRUPT: float = -0.15
W_FAMINE: float = -0.1
W_EXODUS: float = -0.1
W_POSITIVE_INCOME: float = 0.05  # bonus si taxes+passif > maintenance
W_MARBLE_PROGRESS: float = 5.0   # par tranche de 100 marble gagnés (uniquement gain)
W_VICTORY: float = 50.0
W_DEFEAT: float = -10.0
W_SURVIVAL: float = 0.0  # supprimé : évite le plateau DO_NOTHING

# Milestones one-shot (bonus uniques, déclenchés à la première occurrence)
W_FIRST_HOUSE: float = 1.0
W_FIRST_FARM: float = 2.0
W_FIRST_WELL: float = 1.0
W_FIRST_GRANARY: float = 3.0
W_FIRST_MARKET: float = 3.0
W_FIRST_LUMBER_CAMP: float = 2.0
W_FIRST_TRADING_POST: float = 3.0
W_FIRST_POPULATION: float = 5.0
W_FIRST_TEMPLE: float = 8.0
W_FIRST_MARBLE_QUARRY: float = 3.0
W_FIRST_WAREHOUSE_MARBLE: float = 3.0
W_FIRST_BATHS: float = 3.0
W_FIRST_THEATER: float = 3.0
W_BUILD_FORUM: float = 10.0
W_BUILD_PREFECTURE: float = 15.0
W_BUILD_OBELISK: float = 20.0


def compute_reward(
    prev: RewardState,
    curr: RewardState,
    result: TurnResult,
) -> float:
    """Calcule le reward d'un tour selon la formule de shaping.

    Fonction pure : aucun effet de bord, aucune dépendance à GameState.

    Args:
        prev: Snapshot RewardState avant l'action (capturé juste avant step).
        curr: Snapshot RewardState après l'action (capturé juste après step).
        result: TurnResult retourné par engine step, portant les flags du tour.

    Returns:
        Reward scalaire (non clampé).
    """
    reward = 0.0

    # Deltas continus
    reward += W_POP * (curr.total_population - prev.total_population) / 100.0
    reward += W_LEVEL * (curr.city_level - prev.city_level)
    reward += W_SAT * (curr.global_satisfaction - prev.global_satisfaction)
    reward += W_HOUSING * (curr.housing_sum - prev.housing_sum) / 10.0

    # Milestones one-shot : déclenchés uniquement à la première transition False→True
    if not prev.first_house_placed and curr.first_house_placed:
        reward += W_FIRST_HOUSE
    if not prev.first_farm_placed and curr.first_farm_placed:
        reward += W_FIRST_FARM
    if not prev.first_well_placed and curr.first_well_placed:
        reward += W_FIRST_WELL
    if not prev.first_granary_placed and curr.first_granary_placed:
        reward += W_FIRST_GRANARY
    if not prev.first_market_placed and curr.first_market_placed:
        reward += W_FIRST_MARKET
    if not prev.first_lumber_camp_placed and curr.first_lumber_camp_placed:
        reward += W_FIRST_LUMBER_CAMP
    if not prev.first_trading_post_placed and curr.first_trading_post_placed:
        reward += W_FIRST_TRADING_POST
    if not prev.first_population and curr.first_population:
        reward += W_FIRST_POPULATION
    if not prev.first_temple_placed and curr.first_temple_placed:
        reward += W_FIRST_TEMPLE
    if not prev.first_marble_quarry_placed and curr.first_marble_quarry_placed:
        reward += W_FIRST_MARBLE_QUARRY
    if not prev.first_warehouse_marble_placed and curr.first_warehouse_marble_placed:
        reward += W_FIRST_WAREHOUSE_MARBLE
    if not prev.first_baths_placed and curr.first_baths_placed:
        reward += W_FIRST_BATHS
    if not prev.first_theater_placed and curr.first_theater_placed:
        reward += W_FIRST_THEATER
    if not prev.has_forum and curr.has_forum:
        reward += W_BUILD_FORUM
    if not prev.has_prefecture and curr.has_prefecture:
        reward += W_BUILD_PREFECTURE
    if not prev.has_obelisk and curr.has_obelisk:
        reward += W_BUILD_OBELISK

    # Pénalités
    if result.bankrupt:
        reward += W_BANKRUPT
    if result.famine_count > 0:
        reward += W_FAMINE
    if result.exodus > 0:
        reward += W_EXODUS

    # Reward continu marble : récompense uniquement le gain (pas la dépense)
    marble_gain = max(0, curr.marble_stock - prev.marble_stock)
    reward += W_MARBLE_PROGRESS * marble_gain / 100.0

    # Bonus économie viable
    if result.taxes_collected + result.passive_income > result.maintenance_paid:
        reward += W_POSITIVE_INCOME

    # Terminaison
    if result.victory:
        reward += W_VICTORY
    if result.defeat:
        reward += W_DEFEAT

    reward += W_SURVIVAL
    return float(reward)
