"""5 niveaux de ville, conditions de victoire et de défaite."""

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
