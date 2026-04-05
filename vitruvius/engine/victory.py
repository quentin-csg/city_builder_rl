"""5 niveaux de ville, conditions de victoire et de défaite."""

from collections import Counter

from pydantic import BaseModel, field_validator, model_validator


class CityLevelConfig(BaseModel):
    """Configuration d'un niveau de ville."""

    level: int
    id: str
    display_name: str
    min_population: int
    min_satisfaction: float
    required_buildings: list[str]

    @field_validator("min_satisfaction")
    @classmethod
    def satisfaction_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"min_satisfaction doit être entre 0.0 et 1.0, reçu {v}.")
        return v


class CityLevelsConfig(BaseModel):
    """Configuration des niveaux de ville chargée depuis city_levels.yaml."""

    city_levels: list[CityLevelConfig]

    @model_validator(mode="after")
    def validate_ordering(self) -> "CityLevelsConfig":
        levels = [cl.level for cl in self.city_levels]
        if sorted(levels) != list(range(1, len(levels) + 1)):
            raise ValueError("Les niveaux de ville doivent être 1..N sans trous.")
        pops = [cl.min_population for cl in sorted(self.city_levels, key=lambda c: c.level)]
        if pops != sorted(pops):
            raise ValueError("Les populations minimales doivent croître avec le niveau.")
        return self


# ---------------------------------------------------------------------------
# Runtime — évaluation victoire / défaite
# ---------------------------------------------------------------------------


def compute_city_level(
    total_pop: int,
    global_sat: float,
    placed_ids: Counter[str],
    city_levels: list[CityLevelConfig],
) -> int:
    """Retourne le niveau de ville actuel (1–5).

    Itère les niveaux du plus haut au plus bas. Retourne le premier niveau
    dont toutes les conditions sont remplies. Minimum garanti : 1.

    Args:
        total_pop: Population totale actuelle.
        global_sat: Satisfaction globale actuelle (0.0–1.0).
        placed_ids: Compteur des bâtiments posés (Grid._placed_ids).
        city_levels: Liste des configs de niveaux de ville.

    Returns:
        Niveau de ville courant, entier entre 1 et 5.
    """
    for cl in sorted(city_levels, key=lambda c: c.level, reverse=True):
        if cl.level == 1:
            # Le niveau 1 est le minimum absolu — toujours atteint.
            return 1
        if (
            total_pop >= cl.min_population
            and global_sat >= cl.min_satisfaction
            and all(placed_ids[bid] > 0 for bid in cl.required_buildings)
        ):
            return cl.level
    return 1


def check_defeat(
    total_pop: int,
    consecutive_bankrupt_turns: int,
    has_housing: bool = True,
) -> bool:
    """Vérifie les conditions de défaite.

    Args:
        total_pop: Population totale actuelle.
        consecutive_bankrupt_turns: Nombre de tours consécutifs avec denarii < -500.
        has_housing: Au moins une maison est posée sur la grille. Si False,
            pop=0 n'est pas une défaite (ville sans logement = départ normal).

    Returns:
        True si défaite (population nulle avec logement OU banqueroute ≥ 5 tours).
    """
    pop_defeat = total_pop == 0 and has_housing
    return pop_defeat or consecutive_bankrupt_turns >= 5
