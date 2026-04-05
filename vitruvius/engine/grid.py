"""Grille 32×32, terrain, placement de bâtiments."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from vitruvius.engine.buildings import BuildingConfig
from vitruvius.engine.terrain import TerrainType, generate_terrain

# Caractères ASCII par type de terrain
_TERRAIN_CHAR: dict[TerrainType, str] = {
    TerrainType.PLAIN: ".",
    TerrainType.FOREST: "T",
    TerrainType.HILL: "^",
    TerrainType.WATER: "~",
    TerrainType.MARSH: "%",
}


@dataclass
class PlacedBuilding:
    """Bâtiment posé sur la grille."""

    building_id: str
    x: int  # coin haut-gauche
    y: int
    size: tuple[int, int]  # (largeur, hauteur)


class Grid:
    """Grille 32×32 contenant le terrain et les bâtiments placés."""

    SIZE: int = 32

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.terrain: list[list[TerrainType]] = generate_terrain(self.SIZE, seed)
        # Cache des tiles WATER (terrain fixe) — évite un scan O(SIZE²) répété
        self.water_tiles: frozenset[tuple[int, int]] = frozenset(
            (x, y)
            for y in range(self.SIZE)
            for x in range(self.SIZE)
            if self.terrain[y][x] == TerrainType.WATER
        )
        # Chaque case stocke (ox, oy) du coin haut-gauche du bâtiment, ou None
        self._origin: list[list[tuple[int, int] | None]] = [
            [None] * self.SIZE for _ in range(self.SIZE)
        ]
        self.placed_buildings: dict[tuple[int, int], PlacedBuilding] = {}
        # Index building_id → nombre d'instances posées (O(1) pour unicité)
        self._placed_ids: Counter[str] = Counter()

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------

    def get_building_at(self, x: int, y: int) -> PlacedBuilding | None:
        """Retourne le bâtiment occupant la case (x, y), ou None."""
        origin = self._origin[y][x]
        if origin is None:
            return None
        return self.placed_buildings.get(origin)

    # ------------------------------------------------------------------
    # Validation de placement
    # ------------------------------------------------------------------

    def can_place(self, building_id: str, x: int, y: int, config: BuildingConfig) -> bool:
        """Vérifie si un bâtiment peut être placé avec son coin haut-gauche en (x, y).

        Args:
            building_id: Identifiant du bâtiment (clé buildings.yaml).
            x: Colonne du coin haut-gauche.
            y: Ligne du coin haut-gauche.
            config: Configuration du bâtiment.

        Returns:
            True si le placement est légal.
        """
        w, h = config.size

        # 1. Dans les limites
        if x < 0 or y < 0 or x + w > self.SIZE or y + h > self.SIZE:
            return False

        # 2. Cases libres + terrain valide
        for dy in range(h):
            for dx in range(w):
                if self._origin[y + dy][x + dx] is not None:
                    return False
                if self.terrain[y + dy][x + dx] == TerrainType.WATER:
                    return False

        # 3. Contrainte de terrain
        if config.terrain_constraint is not None:
            required = config.terrain_constraint.terrain
            ctype = config.terrain_constraint.type

            if ctype == "all_tiles":
                for dy in range(h):
                    for dx in range(w):
                        if self.terrain[y + dy][x + dx] != required:
                            return False

            elif ctype == "adjacent":
                found = False
                # Cases adjacentes à la zone (4-connexité, hors zone)
                zone = {(x + dx, y + dy) for dy in range(h) for dx in range(w)}
                for dy in range(h):
                    for dx in range(w):
                        cx, cy = x + dx, y + dy
                        for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
                            if (nx, ny) in zone:
                                continue
                            if 0 <= nx < self.SIZE and 0 <= ny < self.SIZE:
                                if self.terrain[ny][nx] == required:
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    return False

        # 4. Unicité
        if config.unique and building_id in self._placed_ids:
            return False

        return True

    # ------------------------------------------------------------------
    # Modification
    # ------------------------------------------------------------------

    def place_building(self, building_id: str, x: int, y: int, config: BuildingConfig) -> None:
        """Place un bâtiment. Suppose can_place() == True."""
        w, h = config.size
        pb = PlacedBuilding(building_id=building_id, x=x, y=y, size=config.size)
        self.placed_buildings[(x, y)] = pb
        self._placed_ids[building_id] += 1
        for dy in range(h):
            for dx in range(w):
                self._origin[y + dy][x + dx] = (x, y)

    def remove_building(self, x: int, y: int) -> PlacedBuilding | None:
        """Supprime le bâtiment dont la case (x, y) fait partie.

        Returns:
            Le PlacedBuilding supprimé, ou None si la case était vide.
        """
        origin = self._origin[y][x]
        if origin is None:
            return None
        pb = self.placed_buildings.pop(origin)
        self._placed_ids[pb.building_id] -= 1
        if self._placed_ids[pb.building_id] == 0:
            del self._placed_ids[pb.building_id]
        ox, oy = origin
        w, h = pb.size
        for dy in range(h):
            for dx in range(w):
                self._origin[oy + dy][ox + dx] = None
        return pb

    # ------------------------------------------------------------------
    # Affichage
    # ------------------------------------------------------------------

    def to_ascii(self) -> str:
        """Retourne une représentation ASCII de la grille (32 lignes)."""
        lines: list[str] = []
        for y in range(self.SIZE):
            row_chars: list[str] = []
            for x in range(self.SIZE):
                origin = self._origin[y][x]
                if origin is not None:
                    pb = self.placed_buildings[origin]
                    bid = pb.building_id
                    char = "#" if bid == "road" else bid[0].upper()
                else:
                    char = _TERRAIN_CHAR[self.terrain[y][x]]
                row_chars.append(char)
            lines.append("".join(row_chars))
        return "\n".join(lines)
