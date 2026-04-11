# Vitruvius

> City builder tour par tour sur le thème de la **Rome antique**, piloté par un agent de **Reinforcement Learning**.

Le joueur — ou un agent **MaskablePPO** — bâtit une colonie romaine case par case sur une grille générée procéduralement. L'objectif : faire progresser la ville du simple campement jusqu'au rang de **Nova Roma** en gérant ressources, besoins des citoyens et événements aléatoires.

Le moteur de simulation est en **Python pur** (headless). Le rendu visuel est optionnel et délégué à **Godot 4** via WebSocket.

---

## Stack

| Composant | Technologie |
| --- | --- |
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

- **Observation** : grille `(32, 32, 31)` (terrain + one-hot 20 bâtiments + niveau maison + 6 services + pop + aqueduct + famine) + features globales `(18,)` — ressources, satisfaction, dynamics inter-tour
- **Espace d'actions** : `Discrete(21505)` — placement (21 bâtiments × 1024 cases), démolition, DO_NOTHING
- **Masque d'actions** dynamique : invalide automatiquement les placements illégaux (terrain, ressources, unicité)
- **Reward shaping** : Δpopulation, Δcity_level, Δsatisfaction, Δhousing + pénalités bankrupt/famine/exode + bonus victoire/défaite
- Passe `check_env()` de Stable-Baselines3 sans erreur

### Bridge WebSocket (`vitruvius/bridge/`)

- **Serveur** (`server.py`) : WebSocket port 9876, une session de jeu par client, multi-client
- **Protocole** (`protocol.py`) : messages JSON typés — `init`, `state`, `ack`, `error` (serveur → client) ; `action`, `reset`, `load_model`, `auto_step` (client → serveur)
- **Mode replay** : `auto_step` charge un modèle MaskablePPO `.zip` et avance de N tours avec dynamics alignées sur l'environnement d'entraînement

### Rendu Godot (`godot/`)

Client minimal (Option A) : connexion WebSocket, affichage log textuel des messages `init`/`state`, bouton `do_nothing`. Nécessite Godot 4.3+.

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
pip install -r requirements.txt

# Lancer tous les tests
pytest -v

# Jouer en terminal
python -m vitruvius.cli --seed 42
```

---

## Tests

570 tests, organisés par module :

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
pytest tests/test_bridge.py           # serveur WebSocket (GameSession, auto_step)
pytest tests/test_protocol.py         # parsing messages JSON client → serveur
```

---

## Structure

```text
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
│   └── train.py
├── bridge/
│   ├── server.py    # WebSocket port 9876
│   └── protocol.py  # parsing / sérialisation messages JSON
godot/               # Client Godot 4 (Option A : texte minimal)
saves/               # Sauvegardes CLI (ignoré par git)
tests/               # Tests pytest par module
```

---

## Roadmap

| Étape | Statut | Description |
| --- | --- | --- |
| 1–13 | ✅ Terminé | Moteur complet, wrapper RL, reward shaping |
| 14–15 | ✅ Terminé | `train.py` — curriculum MaskablePPO |
| 16 | ✅ Terminé | `bridge/server.py` + `protocol.py` — WebSocket (570 tests) |
| 17A | ✅ Terminé | Client Godot minimal (texte + bouton, à tester avec Godot 4.3+) |
| 17B | À faire | Grille terrain, boutons place/demolish, panneau ressources |

---

## Conditions de victoire / défaite

**Victoire** — Nova Roma : 2 500 habitants, 65 % de satisfaction, obélisque + préfecture présents.

**Défaite** — population à 0, ou denarii < -500 pendant 5 tours consécutifs.
