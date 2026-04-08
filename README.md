# Vitruvius

> City builder tour par tour sur le thème de la **Rome antique**, piloté par un agent de **Reinforcement Learning**.

Le joueur — ou un agent **MaskablePPO** — bâtit une colonie romaine case par case sur une grille générée procéduralement. L'objectif : faire progresser la ville du simple campement jusqu'au rang de **Nova Roma** en gérant ressources, besoins des citoyens et événements aléatoires.

Le moteur de simulation est en **Python pur** (headless). Le rendu visuel est optionnel et délégué à **Godot 4** via WebSocket.

---

## Stack

| Composant | Technologie |
|---|---|
| Moteur de simulation | Python 3.11+ |
| Wrapper RL | Gymnasium + sb3-contrib (MaskablePPO) |
| Config gameplay | YAML (Pydantic) |
| Rendu (optionnel) | Godot 4 — isométrique, WebSocket |

---

## Fonctionnalités implémentées

### Moteur de jeu (`vitruvius/engine/`)

- **Grille 32×32** avec génération procédurale de terrain (plaine, forêt, colline, eau, marais)
- **20 bâtiments** avec coûts, maintenance, contraintes de terrain et effets spéciaux
- **Système de ressources** : denarii, blé, bois, marbre — production, stockage limité, taxes, entretien
- **Population** : maisons à 6 niveaux, croissance/régression, famine (FIFO), exode, immigration
- **Services** : eau (aqueduc + fontaine), nourriture (marché), hygiène, religion, divertissement, sécurité — chacun avec son rayon d'influence
- **Événements aléatoires** : incendie, sécheresse, bonne récolte, immigration (tirage cumulatif par tour)
- **Boucle de tour en 13 étapes** : action → production → taxes → famine → services → satisfaction → évolution → croissance → événements → victoire
- **Sérialisation JSON** complète du `GameState` (déterministe via seed RNG)

### Interface RL (`vitruvius/rl/`)

- **Observation** : grille `(32, 32, 12)` + features globales `(15,)` — terrain, bâtiments, couverture services, population, satisfaction, événements actifs
- **Espace d'actions** : `Discrete(21505)` — placement (21 bâtiments × 1024 cases), démolition, DO_NOTHING
- **Masque d'actions** dynamique : invalide automatiquement les placements illégaux (terrain, ressources, unicité)
- **Reward shaping** : Δpopulation, Δcity_level, Δsatisfaction, Δhousing + pénalités bankrupt/famine/exode + bonus victoire/défaite
- Passe `check_env()` de Stable-Baselines3 sans erreur

### CLI (`vitruvius/cli.py`)

Jouer manuellement en terminal :

```bash
python -m vitruvius.cli           # nouvelle partie
python -m vitruvius.cli --seed 42 # partie reproductible
```

Commandes disponibles : `place`, `demolish`, `do_nothing`, `inspect`, `list`, `info`, `save`, `load`, `help`, `quit`.

Les sauvegardes sont stockées dans le dossier `saves/`.

---

## Démarrage rapide

```bash
# Installer les dépendances
pip install -e ".[dev]"

# Lancer tous les tests
pytest -v

# Jouer en terminal
python -m vitruvius.cli --seed 42
```

---

## Tests

470 tests, organisés par module :

```bash
pytest tests/test_engine_grid.py      # grille et placement
pytest tests/test_resources.py        # production, stockage, taxes
pytest tests/test_buildings.py        # placement, démolition, aqueducs
pytest tests/test_population.py       # croissance, famine, exode
pytest tests/test_events.py           # incendie, sécheresse, immigration
pytest tests/test_turn.py             # boucle 13 étapes
pytest tests/test_gym_env.py          # wrapper Gymnasium
pytest tests/test_reward.py           # reward shaping
pytest tests/test_integration.py      # fuzzing multi-seed, invariants
```

---

## Structure

```
vitruvius/
├── engine/          # Simulation pure — aucune dépendance graphique
│   ├── grid.py, terrain.py, buildings.py, resources.py
│   ├── population.py, services.py, events.py
│   ├── turn.py, game_state.py, victory.py, actions.py
├── config/          # Source de vérité gameplay (YAML)
│   ├── buildings.yaml, resources.yaml, needs.yaml
│   ├── events.yaml, city_levels.yaml
├── rl/              # Wrapper Gymnasium + RL
│   ├── gym_env.py, observation.py, action_mask.py, reward.py
│   └── train.py    # (à venir — curriculum MaskablePPO)
├── bridge/
│   └── server.py   # (à venir — WebSocket → Godot)
saves/               # Sauvegardes CLI (ignoré par git)
tests/               # Tests pytest par module
```

---

## Roadmap

| Étape | Statut | Description |
|---|---|---|
| 1–13 | ✅ Terminé | Moteur complet, wrapper RL, reward shaping (470 tests) |
| 14–15 | En cours | `train.py` — curriculum Micro → Petite → Complète |
| 16 | À faire | `bridge/server.py` — WebSocket pour Godot |
| 17 | À faire | Rendu isométrique Godot 4 |

---

## Conditions de victoire / défaite

**Victoire** — Nova Roma : 2 500 habitants, 65 % de satisfaction, obélisque + préfecture présents.

**Défaite** — population à 0, ou denarii < -500 pendant 5 tours consécutifs.
