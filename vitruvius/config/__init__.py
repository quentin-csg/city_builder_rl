"""Chargement centralisé des fichiers YAML de configuration du jeu."""

from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator

from vitruvius.engine.buildings import BuildingsConfig
from vitruvius.engine.events import EventsConfig
from vitruvius.engine.population import NeedsConfig
from vitruvius.engine.resources import ResourcesConfig
from vitruvius.engine.victory import CityLevelsConfig

_DEFAULT_CONFIG_DIR = Path(__file__).parent

_VALID_NEED_TYPES = frozenset(["water", "food", "religion", "hygiene", "entertainment", "security"])


class GameConfig(BaseModel):
    """Configuration agrégée du jeu, chargée depuis les 5 fichiers YAML."""

    resources: ResourcesConfig
    buildings: BuildingsConfig
    needs: NeedsConfig
    events: EventsConfig
    city_levels: CityLevelsConfig

    @model_validator(mode="after")
    def cross_validate(self) -> "GameConfig":
        building_ids = set(self.buildings.buildings.keys())

        # Les storage_buildings référencés dans resources doivent exister
        for res_id, res in self.resources.resources.items():
            if res.storage_building is not None and res.storage_building not in building_ids:
                raise ValueError(
                    f"La ressource '{res_id}' référence un bâtiment de stockage inconnu : "
                    f"'{res.storage_building}'."
                )

        # Les required_buildings des niveaux de ville doivent exister
        for city_level in self.city_levels.city_levels:
            for building_id in city_level.required_buildings:
                if building_id not in building_ids:
                    raise ValueError(
                        f"Le niveau de ville '{city_level.id}' référence un bâtiment inconnu : "
                        f"'{building_id}'."
                    )

        # Les types de service des bâtiments doivent correspondre à des besoins connus
        for bld_id, bld in self.buildings.buildings.items():
            if bld.service is not None and bld.service.type not in _VALID_NEED_TYPES:
                raise ValueError(
                    f"Le bâtiment '{bld_id}' a un service de type inconnu : '{bld.service.type}'."
                )

        return self


def load_config(config_dir: Path | None = None) -> GameConfig:
    """Charge tous les fichiers YAML et retourne un GameConfig validé.

    Args:
        config_dir: Répertoire contenant les fichiers YAML. Par défaut, le répertoire
            du module (vitruvius/config/).

    Returns:
        GameConfig validé avec toutes les données de gameplay.

    Raises:
        FileNotFoundError: Si un fichier YAML est manquant.
        pydantic.ValidationError: Si les données ne respectent pas le schéma attendu.
    """
    config_dir = config_dir or _DEFAULT_CONFIG_DIR

    def _load_yaml(filename: str) -> dict:
        path = config_dir / filename
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    return GameConfig(
        resources=ResourcesConfig(**_load_yaml("resources.yaml")),
        buildings=BuildingsConfig(**_load_yaml("buildings.yaml")),
        needs=NeedsConfig(**_load_yaml("needs.yaml")),
        events=EventsConfig(**_load_yaml("events.yaml")),
        city_levels=CityLevelsConfig(**_load_yaml("city_levels.yaml")),
    )
