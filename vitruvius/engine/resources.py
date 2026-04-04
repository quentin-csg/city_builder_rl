"""Gestion des 4 ressources : Denarii, Blé, Bois, Marbre."""

from pydantic import BaseModel, field_validator


class ResourceConfig(BaseModel):
    """Configuration d'une ressource."""

    display_name: str
    starting_amount: int
    storage_building: str | None
    max_storage: int | None


class PassiveIncomeConfig(BaseModel):
    """Revenu passif par tour, indépendant des bâtiments."""

    denarii: int

    @field_validator("denarii")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("Le revenu passif en denarii ne peut pas être négatif.")
        return v


class ResourcesConfig(BaseModel):
    """Configuration complète des ressources chargée depuis resources.yaml."""

    resources: dict[str, ResourceConfig]
    passive_income: PassiveIncomeConfig
