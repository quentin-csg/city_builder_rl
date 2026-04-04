# Vitruvius

City builder tour par tour sur le thème de la **Rome antique**, entraîné par un agent de **Reinforcement Learning**.

## Idée principale

Le joueur (ou un agent RL) construit une colonie romaine case par case sur une grille générée procéduralement. L'objectif est de faire progresser la ville du rang de simple campement jusqu'au niveau **Nova Roma** — la cité impériale — en gérant ressources, besoins des citoyens et événements aléatoires.

Le moteur de simulation est en **Python pur** (headless). Un agent **MaskablePPO** (Stable-Baselines3) apprend à jouer de manière autonome. Le rendu visuel est optionnel et délégué à **Godot 4** via WebSocket.

## Stack

- **Python 3.11+** — moteur de simulation + entraînement RL
- **Gymnasium + sb3-contrib** — wrapper et MaskablePPO
- **Godot 4** — rendu isométrique (optionnel, non requis pour l'entraînement)

## Lancer les tests

```bash
pytest -v
```

## Structure

```text
vitruvius/
├── engine/   # Simulation pure (grille, bâtiments, population, événements)
├── rl/       # Wrapper Gymnasium, reward shaping, entraînement
├── config/   # Données gameplay en YAML
└── bridge/   # Serveur WebSocket pour Godot
tests/        # Tests pytest par module
cli.py        # Interface terminal pour tester manuellement
```
