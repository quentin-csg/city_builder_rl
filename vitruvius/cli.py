"""CLI interactive pour jouer au city builder en terminal.

Utilisation :
    python -m vitruvius.cli [--seed N]

Commandes :
    place <id> <x> <y>   (alias: p)   Placer un bâtiment
    demolish <x> <y>     (alias: d)   Démolir le bâtiment en (x,y)
    wait                 (alias: w)   Passer le tour (do_nothing)
    inspect <x> <y>      (alias: x)   Inspecter la case (terrain, bâtiment, couverture)
    list                 (alias: ls)  Lister les bâtiments disponibles
    info <id>            (alias: i)   Détails d'un bâtiment
    save <fichier>                    Sauvegarder dans saves/<fichier>
    load <fichier>                    Charger depuis saves/<fichier>
    help                 (alias: h ?) Afficher cette aide
    quit                 (alias: q)   Quitter
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from vitruvius.config import load_config
from vitruvius.engine.game_state import from_dict, init_game_state, to_dict
from vitruvius.engine.services import compute_coverage
from vitruvius.engine.terrain import TerrainType
from vitruvius.engine.turn import Action, TurnResult, step

if TYPE_CHECKING:
    from vitruvius.config import GameConfig
    from vitruvius.engine.game_state import GameState
    from vitruvius.engine.grid import Grid


# ---------------------------------------------------------------------------
# Constantes d'affichage
# ---------------------------------------------------------------------------

# Réutilise les mêmes chars que grid.py pour cohérence
_TERRAIN_CHARS: dict[TerrainType, str] = {
    TerrainType.PLAIN:  ".",
    TerrainType.FOREST: "T",
    TerrainType.HILL:   "^",
    TerrainType.WATER:  "~",
    TerrainType.MARSH:  "%",
}

# Un caractère unique par bâtiment
_BUILDING_CHARS: dict[str, str] = {
    "road":             "+",
    "well":             "w",
    "fountain":         "f",
    "aqueduct":         "a",
    "housing":          "H",
    "wheat_farm":       "W",
    "lumber_camp":      "L",
    "marble_quarry":    "Q",
    "granary":          "G",
    "market":           "M",
    "small_altar":      "s",
    "temple":           "P",
    "baths":            "b",
    "theater":          "t",
    "forum":            "F",
    "prefecture":       "X",
    "trading_post":     "R",
    "warehouse_wood":   "o",
    "warehouse_marble": "O",
    "obelisk":          "*",
}

_HELP_TEXT = __doc__ or ""


# ---------------------------------------------------------------------------
# Rendu
# ---------------------------------------------------------------------------


def render_grid(grid: Grid) -> str:
    """Retourne une représentation ASCII de la grille 32×32.

    Chaque case est un caractère :
    - Bâtiment : caractère spécifique par type (_BUILDING_CHARS)
    - Terrain libre : caractère par type de terrain (_TERRAIN_CHARS)

    Args:
        grid: Grille de jeu.

    Returns:
        Chaîne multi-lignes avec en-têtes de colonnes et numéros de lignes.
    """
    lines: list[str] = []

    # En-tête colonne : graduations de 5 en 5 (coordonnées 1-indexées)
    header_tens = "   " + "".join(
        str(x // 10) if x % 5 == 0 else " " for x in range(1, grid.SIZE + 1)
    )
    header_units = "   " + "".join(
        str(x % 10) if x % 5 == 0 else " " for x in range(1, grid.SIZE + 1)
    )
    lines.append(header_tens)
    lines.append(header_units)

    for y in range(grid.SIZE):
        row_chars = []
        for x in range(grid.SIZE):
            pb = grid.get_building_at(x, y)
            if pb is not None:
                row_chars.append(_BUILDING_CHARS.get(pb.building_id, "?"))
            else:
                row_chars.append(_TERRAIN_CHARS.get(grid.terrain[y][x], "?"))
        lines.append(f"{y + 1:2d} {''.join(row_chars)}")

    return "\n".join(lines)


def render_state(gs: GameState) -> str:
    """Retourne un résumé textuel de l'état global de la partie.

    Args:
        gs: État courant du jeu.

    Returns:
        Chaîne sur 3 lignes résumant tour, ressources et événements.
    """
    rs = gs.resource_state
    events_str = ", ".join(
        f"{e.event_type}({e.turns_remaining})" for e in gs.active_events
    ) or "aucun"

    pop = sum(h.population for h in gs.houses.values())

    return (
        f"Tour: {gs.turn} | Niveau: {gs.city_level} | "
        f"Satisfaction: {gs.global_satisfaction:.0%} | Pop: {pop}\n"
        f"Denarii: {rs.denarii:.2f} | Blé: {rs.wheat} | "
        f"Bois: {rs.wood} | Marbre: {rs.marble}\n"
        f"Maisons: {len(gs.houses)} | Événements: {events_str}"
    )


def format_turn_result(r: TurnResult) -> str:
    """Retourne un résumé textuel du tour qui vient de se passer.

    Args:
        r: Résultat du tour.

    Returns:
        Chaîne multi-lignes décrivant les effets du tour.
    """
    parts: list[str] = []

    if not r.action_succeeded:
        parts.append("Action impossible (ressources insuffisantes, terrain invalide ou case occupée)")

    if r.production:
        prod_str = ", ".join(f"+{v} {k}" for k, v in r.production.items() if v > 0)
        if prod_str:
            parts.append(f"Production : {prod_str}")

    parts.append(
        f"Taxes : +{r.taxes_collected:.2f} | Maintenance : -{r.maintenance_paid:.2f}"
    )

    if r.famine_count > 0:
        parts.append(f"Famine : {r.famine_count} maison(s), -{r.famine_pop_lost} hab")

    if r.growth > 0:
        parts.append(f"Croissance : +{r.growth} hab")
    if r.exodus > 0:
        parts.append(f"Exode : -{r.exodus} hab")
    if r.evolved > 0:
        parts.append(f"Évolutions : {r.evolved}")
    if r.regressed > 0:
        parts.append(f"Régressions : {r.regressed}")
    if r.new_event is not None:
        parts.append(f"Événement : {r.new_event.event_type}")
    if r.bankrupt:
        parts.append("⚠ Banqueroute !")

    return "\n".join(parts) if parts else "(aucun changement notable)"


def format_buildings_list(config: GameConfig) -> str:
    """Retourne un tableau listé des 20 bâtiments avec coût et taille.

    Args:
        config: Configuration du jeu.

    Returns:
        Chaîne tabulée.
    """
    lines = [f"{'ID':<20} {'Taille':<8} {'Coût'}"]
    lines.append("-" * 50)
    for bid, cfg in config.buildings.buildings.items():
        size_str = f"{cfg.size[0]}x{cfg.size[1]}"
        cost_parts = [f"{v} {k}" for k, v in cfg.cost.items()]
        cost_str = ", ".join(cost_parts) if cost_parts else "gratuit"
        lines.append(f"{bid:<20} {size_str:<8} {cost_str}")
    return "\n".join(lines)


def format_building_info(building_id: str, config: GameConfig) -> str:
    """Retourne les détails d'un bâtiment.

    Args:
        building_id: Identifiant du bâtiment.
        config: Configuration du jeu.

    Returns:
        Chaîne descriptive.

    Raises:
        KeyError: Si building_id inconnu.
    """
    cfg = config.buildings.buildings[building_id]
    lines = [f"=== {cfg.display_name} ({building_id}) ==="]
    lines.append(f"Taille : {cfg.size[0]}x{cfg.size[1]}")

    cost_parts = [f"{v} {k}" for k, v in cfg.cost.items()]
    lines.append(f"Coût : {', '.join(cost_parts) if cost_parts else 'gratuit'}")

    if cfg.production:
        lines.append(f"Production : {cfg.production.amount} {cfg.production.resource}/tour")

    if cfg.service:
        lines.append(f"Service : {cfg.service.need_covered}, rayon={cfg.service.radius}")

    if cfg.special_effect:
        se = cfg.special_effect
        if se.tax_bonus:
            lines.append(f"Effet spécial : bonus taxes +{se.tax_bonus:.0%}")
        if se.is_housing:
            lines.append("Effet spécial : logement")
        if se.requires_aqueduct:
            lines.append("Effet spécial : requiert aqueduc")

    if cfg.unique:
        lines.append("Unique : oui (une seule instance possible)")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Inspection de case
# ---------------------------------------------------------------------------


def format_inspect(x: int, y: int, gs: GameState, config: GameConfig) -> str:
    """Retourne le détail complet d'une case (x, y).

    Affiche le terrain, le bâtiment éventuel et — pour les maisons —
    la population, le niveau, le statut famine et la couverture de service.

    Args:
        x: Colonne (0–31).
        y: Ligne (0–31).
        gs: État courant du jeu.
        config: Configuration du jeu.

    Returns:
        Chaîne multi-lignes décrivant la case.
    """
    grid = gs.grid
    lines = [f"=== Case ({x + 1}, {y + 1}) ==="]
    lines.append(f"Terrain : {grid.terrain[y][x].name.lower()}")

    pb = grid.get_building_at(x, y)
    if pb is None:
        lines.append("Bâtiment : aucun")
        return "\n".join(lines)

    cfg = config.buildings.buildings[pb.building_id]
    lines.append(f"Bâtiment : {cfg.display_name} ({pb.building_id})")
    lines.append(f"  Origine : ({pb.x + 1}, {pb.y + 1}) | Taille : {pb.size[0]}x{pb.size[1]}")

    # Détail maison
    origin = (pb.x, pb.y)
    if origin in gs.houses:
        h = gs.houses[origin]
        house_levels = config.needs.house_levels
        level_cfg = next((hl for hl in house_levels if hl.level == h.level), None)

        lines.append(f"  Niveau : {h.level} ({level_cfg.display_name if level_cfg else '?'})")
        lines.append(f"  Population : {h.population} / {level_cfg.max_population if level_cfg else '?'}")
        lines.append(f"  Famine : {'OUI' if h.famine else 'non'}")

        # Couverture de service pour cette maison
        coverage = compute_coverage(
            grid, config.buildings.buildings, gs.resource_state
        )
        covered = coverage.get(origin, set())

        if level_cfg is not None:
            required = set(level_cfg.required_needs)
            next_cfg = next((hl for hl in house_levels if hl.level == h.level + 1), None)
            next_required = set(next_cfg.required_needs) if next_cfg else set()
        else:
            required = set()
            next_required = set()

        all_needs = {"water", "food", "religion", "hygiene", "entertainment", "security"}
        lines.append("  Besoins (couverture) :")
        for need in sorted(all_needs):
            ok = need in covered
            current = " [requis]" if need in required else ""
            nxt = " [requis niveau+1]" if need in next_required - required else ""
            lines.append(f"    {'OK' if ok else '--'} {need}{current}{nxt}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing des commandes
# ---------------------------------------------------------------------------


def parse_command(line: str, building_list: list[str]) -> Action | str:
    """Parse une ligne de commande utilisateur.

    Args:
        line: Ligne saisie par l'utilisateur.
        building_list: Liste des building_id valides.

    Returns:
        Action si la commande produit une action de jeu,
        ou chaîne de commande spéciale parmi :
        "quit", "help", "list", "info:<id>", "save:<file>", "load:<file>".

    Raises:
        ValueError: Commande invalide ou arguments incorrects.
    """
    parts = line.strip().split()

    # Ligne vide ou "wait" / "w" → do_nothing
    if not parts or parts[0] in ("wait", "w"):
        return Action("do_nothing")

    cmd = parts[0].lower()

    # Quit
    if cmd in ("quit", "q"):
        return "quit"

    # Help
    if cmd in ("help", "h", "?"):
        return "help"

    # List
    if cmd in ("list", "ls"):
        return "list"

    # Inspect
    if cmd in ("inspect", "x"):
        if len(parts) < 3:
            raise ValueError("Usage : inspect <x> <y>")
        try:
            ix, iy = int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError("x et y doivent être des entiers.")
        if not (1 <= ix <= 32 and 1 <= iy <= 32):
            raise ValueError(f"Coordonnées hors grille : x={ix}, y={iy} (doit être 1–32).")
        return f"inspect:{ix - 1},{iy - 1}"

    # Info
    if cmd in ("info", "i"):
        if len(parts) < 2:
            raise ValueError("Usage : info <building_id>")
        return f"info:{parts[1]}"

    # Save
    if cmd == "save":
        if len(parts) < 2:
            raise ValueError("Usage : save <fichier>")
        return f"save:{parts[1]}"

    # Load
    if cmd == "load":
        if len(parts) < 2:
            raise ValueError("Usage : load <fichier>")
        return f"load:{parts[1]}"

    # Place
    if cmd in ("place", "p"):
        if len(parts) < 4:
            raise ValueError("Usage : place <building_id> <x> <y>")
        building_id = parts[1]
        if building_id not in building_list:
            raise ValueError(
                f"Bâtiment inconnu : '{building_id}'. Tapez 'list' pour voir la liste."
            )
        try:
            x, y = int(parts[2]), int(parts[3])
        except ValueError:
            raise ValueError("x et y doivent être des entiers.")
        if not (1 <= x <= 32 and 1 <= y <= 32):
            raise ValueError(f"Coordonnées hors grille : x={x}, y={y} (doit être 1–32).")
        return Action("place", building_id, x - 1, y - 1)

    # Demolish
    if cmd in ("demolish", "d"):
        if len(parts) < 3:
            raise ValueError("Usage : demolish <x> <y>")
        try:
            x, y = int(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError("x et y doivent être des entiers.")
        if not (1 <= x <= 32 and 1 <= y <= 32):
            raise ValueError(f"Coordonnées hors grille : x={x}, y={y} (doit être 1–32).")
        return Action("demolish", x=x - 1, y=y - 1)

    raise ValueError(f"Commande inconnue : '{cmd}'. Tapez 'help' pour l'aide.")


# ---------------------------------------------------------------------------
# Boucle principale
# ---------------------------------------------------------------------------


def play(seed: int = 42) -> None:
    """Boucle interactive principale.

    Args:
        seed: Seed pour la génération du terrain et le RNG.
    """
    config = load_config()
    gs = init_game_state(config, seed=seed)
    building_list = list(config.buildings.buildings.keys())

    print(f"=== Nova Roma City Builder (seed={seed}) ===")
    print("Tapez 'help' pour la liste des commandes.\n")

    while not gs.done:
        print("\n" + render_grid(gs.grid))
        print("\n" + render_state(gs))
        print()

        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nPartie interrompue.")
            break

        try:
            result = parse_command(line, building_list)
        except ValueError as e:
            print(f"Erreur : {e}")
            continue

        # Commandes spéciales
        if result == "quit":
            print("Partie quittée.")
            break
        if result == "help":
            print(_HELP_TEXT)
            continue
        if result == "list":
            print(format_buildings_list(config))
            continue
        if isinstance(result, str) and result.startswith("inspect:"):
            coords = result[8:].split(",")
            ix, iy = int(coords[0]), int(coords[1])
            print(format_inspect(ix, iy, gs, config))
            continue
        if isinstance(result, str) and result.startswith("info:"):
            bid = result[5:]
            if bid not in config.buildings.buildings:
                print(f"Bâtiment inconnu : '{bid}'.")
            else:
                print(format_building_info(bid, config))
            continue
        if isinstance(result, str) and result.startswith("save:"):
            name = result[5:]
            filepath = Path(name) if Path(name).is_absolute() else Path("saves") / name
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(to_dict(gs), f, indent=2)
                print(f"Partie sauvegardée : {filepath}")
            except OSError as e:
                print(f"Erreur de sauvegarde : {e}")
            continue
        if isinstance(result, str) and result.startswith("load:"):
            name = result[5:]
            filepath = Path(name) if Path(name).is_absolute() else Path("saves") / name
            try:
                with open(filepath, encoding="utf-8") as f:
                    data = json.load(f)
                gs = from_dict(data, config)
                print(f"Partie chargée : {filepath}")
            except (OSError, KeyError, ValueError) as e:
                print(f"Erreur de chargement : {e}")
            continue

        # Action de jeu
        if isinstance(result, Action):
            turn_result = step(gs, config, result)
            print("\n--- Résultat du tour ---")
            print(format_turn_result(turn_result))

    # Fin de partie
    if gs.done:
        print("\n" + render_grid(gs.grid))
        print("\n" + render_state(gs))
        if gs.victory:
            print("\nVICTOIRE — Nova Roma est fondée !")
        else:
            print("\nDEFAITE — La ville est tombée.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Point d'entrée CLI. Accepte --seed N en argument."""
    seed = 42
    args = sys.argv[1:]
    if "--seed" in args:
        idx = args.index("--seed")
        try:
            seed = int(args[idx + 1])
        except (IndexError, ValueError):
            print("Usage : python -m vitruvius.cli [--seed N]", file=sys.stderr)
            sys.exit(1)
    play(seed=seed)


if __name__ == "__main__":
    main()
