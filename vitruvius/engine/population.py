"""Croissance, besoins, satisfaction, niveaux de maison."""

from pydantic import BaseModel, model_validator

VALID_NEEDS = frozenset(["water", "food", "religion", "hygiene", "entertainment", "security"])


class HouseLevelConfig(BaseModel):
    """Configuration d'un niveau de maison."""

    level: int
    id: str
    display_name: str
    max_population: int
    required_needs: list[str]
    tax_per_inhabitant: float

    @model_validator(mode="after")
    def validate_needs(self) -> "HouseLevelConfig":
        invalid = set(self.required_needs) - VALID_NEEDS
        if invalid:
            raise ValueError(f"Besoins inconnus pour le niveau {self.level} : {invalid}")
        return self


class NeedsConfig(BaseModel):
    """Configuration des niveaux de maison chargée depuis needs.yaml."""

    house_levels: list[HouseLevelConfig]

    @model_validator(mode="after")
    def validate_levels(self) -> "NeedsConfig":
        levels = [hl.level for hl in self.house_levels]
        if sorted(levels) != list(range(1, len(levels) + 1)):
            raise ValueError(f"Les niveaux de maison doivent être 1..{len(levels)} sans trous.")
        return self
