"""Définitions des 20 bâtiments, validation de placement."""

from typing import Literal

from pydantic import BaseModel

from vitruvius.engine.terrain import TerrainType


class TerrainConstraint(BaseModel):
    """Contrainte de terrain pour le placement d'un bâtiment."""

    type: Literal["all_tiles", "adjacent"]
    terrain: TerrainType


class ProductionConfig(BaseModel):
    """Production de ressource par tour."""

    resource: str
    amount: int


class StorageConfig(BaseModel):
    """Capacité de stockage d'une ressource."""

    resource: str
    capacity: int


class ServiceConfig(BaseModel):
    """Service fourni par un bâtiment dans son rayon d'influence."""

    type: str
    radius: int


class SpecialEffect(BaseModel):
    """Effets spéciaux d'un bâtiment (champs optionnels, tous à leur valeur neutre par défaut)."""

    requires_aqueduct: bool = False
    is_aqueduct: bool = False
    is_housing: bool = False
    tax_bonus: float = 0.0
    fire_risk_divisor: int = 1


class BuildingConfig(BaseModel):
    """Configuration complète d'un bâtiment."""

    display_name: str
    category: str
    size: tuple[int, int]
    cost: dict[str, int]
    maintenance: int
    unique: bool
    terrain_constraint: TerrainConstraint | None = None
    production: ProductionConfig | None = None
    storage: StorageConfig | None = None
    service: ServiceConfig | None = None
    special_effect: SpecialEffect | None = None


class BuildingsConfig(BaseModel):
    """Configuration complète des bâtiments chargée depuis buildings.yaml."""

    buildings: dict[str, BuildingConfig]
