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
    total_houses: int = 0  # nombre total de maisons posées (pour famine proportionnelle)

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
    first_marble_quarry_placed: bool = False
    first_warehouse_marble_placed: bool = False
    first_fountain_placed: bool = False
    first_aqueduct_placed: bool = False
    first_theater_placed: bool = False
    first_baths_placed: bool = False

    # Milestone première population
    first_population: bool = False

    # Paliers de population (one-shot)
    reached_pop_100: bool = False
    reached_pop_250: bool = False
    reached_pop_500: bool = False
    reached_pop_1000: bool = False
    reached_pop_2000: bool = False

    # Paliers d'évolution des maisons (one-shot)
    first_house_level_2: bool = False
    first_house_level_3: bool = False
    first_house_level_4: bool = False
    first_house_level_5: bool = False
    first_house_level_6: bool = False

    # Paliers de marbre (one-shot — évite le reward hacking continu)
    reached_marble_50: bool = False
    reached_marble_100: bool = False
    reached_marble_200: bool = False
    reached_marble_500: bool = False


# ---------------------------------------------------------------------------
# Coefficients continus
# ---------------------------------------------------------------------------

W_POP: float = 5.0        # par tranche de 100 hab gagnés (réduit car milestones guident)
W_SAT: float = 3.0        # par point de satisfaction gagné (0–1)
W_HOUSING: float = 2.0    # par tranche de 10 niveaux de maison gagnés
W_BANKRUPT: float = -0.3
W_FAMINE: float = -0.3    # × famine_ratio (proportionnel, pas binaire)
W_EXODUS: float = -0.1
W_POSITIVE_INCOME: float = 0.15  # bonus si taxes+passif > maintenance
W_VICTORY: float = 200.0
W_DEFEAT: float = -30.0
W_SURVIVAL: float = 0.0  # supprimé : évite le plateau DO_NOTHING

# Reward par niveau de ville (remplace W_LEVEL plat — progression exponentielle)
CITY_LEVEL_REWARDS: dict[int, float] = {
    2: 25.0,
    3: 40.0,
    4: 60.0,
    5: 100.0,
}

# ---------------------------------------------------------------------------
# Milestones bâtiments (one-shot)
# ---------------------------------------------------------------------------

W_FIRST_HOUSE: float = 1.0
W_FIRST_FARM: float = 2.0
W_FIRST_WELL: float = 1.0
W_FIRST_GRANARY: float = 3.0
W_FIRST_MARKET: float = 3.0
W_FIRST_LUMBER_CAMP: float = 2.0
W_FIRST_TRADING_POST: float = 3.0
W_FIRST_POPULATION: float = 5.0
W_FIRST_TEMPLE: float = 8.0
W_BUILD_FORUM: float = 10.0
W_BUILD_PREFECTURE: float = 15.0
W_BUILD_OBELISK: float = 20.0
W_FIRST_MARBLE_QUARRY: float = 4.0
W_FIRST_WAREHOUSE_MARBLE: float = 3.0
W_FIRST_FOUNTAIN: float = 2.0
W_FIRST_AQUEDUCT: float = 2.0
W_FIRST_THEATER: float = 5.0
W_FIRST_BATHS: float = 5.0

# ---------------------------------------------------------------------------
# Milestones population (one-shot)
# ---------------------------------------------------------------------------

W_POP_100: float = 2.0
W_POP_250: float = 3.0
W_POP_500: float = 5.0
W_POP_1000: float = 8.0
W_POP_2000: float = 12.0

# ---------------------------------------------------------------------------
# Milestones évolution maisons (one-shot)
# ---------------------------------------------------------------------------

W_HOUSE_LEVEL_2: float = 2.0
W_HOUSE_LEVEL_3: float = 3.0
W_HOUSE_LEVEL_4: float = 5.0
W_HOUSE_LEVEL_5: float = 8.0
W_HOUSE_LEVEL_6: float = 10.0

# ---------------------------------------------------------------------------
# Milestones marbre (one-shot — PAS de reward continu)
# ---------------------------------------------------------------------------

W_MARBLE_50: float = 1.0
W_MARBLE_100: float = 2.0
W_MARBLE_200: float = 3.0
W_MARBLE_500: float = 5.0


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

    # --- Deltas continus ---
    reward += W_POP * (curr.total_population - prev.total_population) / 100.0
    reward += W_SAT * (curr.global_satisfaction - prev.global_satisfaction)
    reward += W_HOUSING * (curr.housing_sum - prev.housing_sum) / 10.0

    # --- Reward par niveau de ville (per-level scaling) ---
    if curr.city_level > prev.city_level:
        for lvl in range(prev.city_level + 1, curr.city_level + 1):
            reward += CITY_LEVEL_REWARDS.get(lvl, 0.0)
    elif curr.city_level < prev.city_level:
        for lvl in range(curr.city_level + 1, prev.city_level + 1):
            reward -= CITY_LEVEL_REWARDS.get(lvl, 0.0)

    # --- Milestones bâtiments (False→True uniquement) ---
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
    if not prev.has_forum and curr.has_forum:
        reward += W_BUILD_FORUM
    if not prev.has_prefecture and curr.has_prefecture:
        reward += W_BUILD_PREFECTURE
    if not prev.has_obelisk and curr.has_obelisk:
        reward += W_BUILD_OBELISK
    if not prev.first_marble_quarry_placed and curr.first_marble_quarry_placed:
        reward += W_FIRST_MARBLE_QUARRY
    if not prev.first_warehouse_marble_placed and curr.first_warehouse_marble_placed:
        reward += W_FIRST_WAREHOUSE_MARBLE
    if not prev.first_fountain_placed and curr.first_fountain_placed:
        reward += W_FIRST_FOUNTAIN
    if not prev.first_aqueduct_placed and curr.first_aqueduct_placed:
        reward += W_FIRST_AQUEDUCT
    if not prev.first_theater_placed and curr.first_theater_placed:
        reward += W_FIRST_THEATER
    if not prev.first_baths_placed and curr.first_baths_placed:
        reward += W_FIRST_BATHS

    # --- Milestones population ---
    if not prev.reached_pop_100 and curr.reached_pop_100:
        reward += W_POP_100
    if not prev.reached_pop_250 and curr.reached_pop_250:
        reward += W_POP_250
    if not prev.reached_pop_500 and curr.reached_pop_500:
        reward += W_POP_500
    if not prev.reached_pop_1000 and curr.reached_pop_1000:
        reward += W_POP_1000
    if not prev.reached_pop_2000 and curr.reached_pop_2000:
        reward += W_POP_2000

    # --- Milestones évolution maisons ---
    if not prev.first_house_level_2 and curr.first_house_level_2:
        reward += W_HOUSE_LEVEL_2
    if not prev.first_house_level_3 and curr.first_house_level_3:
        reward += W_HOUSE_LEVEL_3
    if not prev.first_house_level_4 and curr.first_house_level_4:
        reward += W_HOUSE_LEVEL_4
    if not prev.first_house_level_5 and curr.first_house_level_5:
        reward += W_HOUSE_LEVEL_5
    if not prev.first_house_level_6 and curr.first_house_level_6:
        reward += W_HOUSE_LEVEL_6

    # --- Milestones marbre (one-shot, jamais continus) ---
    if not prev.reached_marble_50 and curr.reached_marble_50:
        reward += W_MARBLE_50
    if not prev.reached_marble_100 and curr.reached_marble_100:
        reward += W_MARBLE_100
    if not prev.reached_marble_200 and curr.reached_marble_200:
        reward += W_MARBLE_200
    if not prev.reached_marble_500 and curr.reached_marble_500:
        reward += W_MARBLE_500

    # --- Pénalités ---
    if result.bankrupt:
        reward += W_BANKRUPT
    # Famine proportionnelle : gradient pour amélioration partielle
    if result.famine_count > 0 and curr.total_houses > 0:
        famine_ratio = result.famine_count / curr.total_houses
        reward += W_FAMINE * famine_ratio
    if result.exodus > 0:
        reward += W_EXODUS

    # --- Bonus économie viable ---
    if result.taxes_collected + result.passive_income > result.maintenance_paid:
        reward += W_POSITIVE_INCOME

    # --- Terminaison ---
    if result.victory:
        reward += W_VICTORY
    if result.defeat:
        reward += W_DEFEAT

    reward += W_SURVIVAL
    return float(reward)
