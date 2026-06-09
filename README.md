# Optimisation de l'Ordonnancement d'un Réseau de Capteurs

Ce projet implémente une solution basée sur la recherche opérationnelle pour maximiser la durée de vie d'un réseau de capteurs vidéo sans fil. Il s'appuie sur deux heuristiques de génération de configurations élémentaires couplées à un programme linéaire continu résolu via le solveur GLPK ou scipy.

## Structure du Projet

Le projet est divisé en cinq modules distincts :

### 1. `data_manager.py` (Partie 1 — Manipulation des données)
Gère l'ingestion et la création des instances du problème.
- **`from_file(filepath)`** : Lit une topologie depuis un fichier texte (format : N capteurs, M zones, durées de vie, listes de zones couvertes par capteur).
- **`from_keyboard()`** : Saisie interactive par l'utilisateur.
- **`generate_random()`** : Génère une instance aléatoire pour les tests.
- **`display()`** : Affiche un résumé lisible de l'instance.

### 2. `heuristic.py` (Partie 2 — Construction de configurations élémentaires)
Contient deux heuristiques pour la génération de configurations élémentaires valides :
- **`generate_random_greedy()`** : Génère des couvertures aléatoires gloutonnes, puis les réduit pour les rendre élémentaires. Maximise la diversité de l'espace de recherche.
- **`generate_greedy_max_coverage()`** : Choisit à chaque étape le capteur couvrant le plus de zones non couvertes, avec un bruit aléatoire pour la diversité. Tend à produire des configurations plus compactes.

Une configuration est dite **élémentaire** si le retrait de n'importe quel capteur brise la couverture complète.

### 3. `lp_solver.py` (Partie 3 — Programme linéaire)
Pont entre la logique Python et la résolution mathématique :
- **`generate_lp_file()`** : Écrit le PL au format CPLEX (`.lp`) pour GLPK.
- **`solve()`** : Fonction unifiée qui essaie GLPK d'abord, puis bascule sur `scipy.optimize.linprog` si GLPK n'est pas disponible.
- **`display_solution()`** : Affiche la solution optimale (durée de vie, temps par configuration, bilan énergétique par capteur).

### 4. `main.py` (Script principal)
Point d'entrée interactif ou en ligne de commande :
- Mode interactif : menus pour le choix de la source de données et de l'heuristique.
- Mode CLI : `python main.py <fichier> [num_configs] [method]`

### 5. `analysis.py` (Parties 4 & 5 — Expérimentation et analyse)
Script d'expérimentation automatique :
- **Partie 4** : Résolution des 5 instances de test avec tableau de résultats.
- **Partie 5** : Analyse comparative de l'influence du nombre et du type de configurations sur la durée de vie du réseau, avec génération de graphiques.

## Format des fichiers de données

```
Ligne 1 : nombre de capteurs (N)
Ligne 2 : nombre de zones (M)
Ligne 3 : N durées de vie séparées par des espaces
Lignes 4 à N+3 : zones couvertes par chaque capteur (indices commençant à 1)
```

## Prérequis

- **Python 3.x**
- **scipy** : `pip install scipy`
- **numpy** : `pip install numpy`
- **matplotlib** : `pip install matplotlib`
- **GLPK** (optionnel) : `sudo apt install glpk-utils` — si absent, scipy est utilisé automatiquement.

## Exécution

### Résoudre une instance (interactif)
```bash
python main.py
```

### Résoudre une instance (ligne de commande)
```bash
python main.py fichier-exemple.txt 50 random
python main.py moyen_test_3.txt 100 greedy
```

### Lancer l'analyse complète (Parties 4 & 5)
```bash
python analysis.py
```

## Fichiers de test fournis

| Fichier | Capteurs (N) | Zones (M) |
|---------|:------------:|:---------:|
| `fichier-exemple.txt` | 4 | 3 |
| `moyen_test_2.txt` | 20 | 10 |
| `moyen_test_3.txt` | 10 | 10 |
| `gros_test_1.txt` | 100 | 200 |
| `maxi_test_1.txt` | 1000 | 500 |