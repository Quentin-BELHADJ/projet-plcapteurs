# Optimisation de l'Ordonnancement d'un Réseau de Capteurs

Ce projet implémente une solution basée sur la recherche opérationnelle pour maximiser la durée de vie d'un réseau de capteurs vidéo sans fil. Il s'appuie sur une heuristique de génération de colonnes couplée à un programme linéaire continu résolu via le solveur GLPK.

## Structure du Projet

Le projet est divisé en quatre modules distincts, conçus de manière modulaire :

### 1. `data_manager.py` (Ingestion des données)
Ce module gère l'ingestion et la création des instances du problème.
- **`from_file`** : Permet de lire une topologie de réseau depuis un fichier texte (nombre de zones, capteurs, durées de vie, matrice de couverture).
- **`generate_random`** : Génère une instance aléatoire pour les tests, en garantissant qu'il n'y a pas de "zone morte" (toutes les zones sont couvertes au moins une fois par un capteur).

### 2. `heuristic.py` (Algorithmique)
Ce fichier contient l'algorithme de génération de "configurations élémentaires" valides.
- Une configuration est un sous-ensemble de capteurs suffisant pour surveiller l'intégralité des zones du réseau.
- Elle est dite **élémentaire** si le retrait d'un seul de ses capteurs brise cette couverture globale.
- L'approche implémentée génère d'abord une couverture aléatoire gloutonne (pour maximiser la diversité de l'espace de recherche), puis la réduit en testant chaque capteur pour éliminer la redondance. 

### 3. `lp_solver.py` (Modélisation Linéaire)
C'est le pont entre la logique Python et le solveur mathématique.
- **`generate_lp_file`** : Modélise mathématiquement l'instance sous forme d'un programme linéaire continu (maximisation de la somme des temps d'activation $t_j$) respectant la contrainte de durée de vie de la batterie de chaque capteur. Il sauvegarde le modèle au format standard CPLEX (`.lp`).
- **`solve_lp`** : Automatise l'appel au solveur externe **GLPK** via la ligne de commande (`glpsol`), sans interrompre l'exécution de Python.
- **`parse_solution`** : Lit et décode le fichier résultat généré par GLPK pour en extraire la durée de vie maximale trouvée (la fonction objectif) et le détail des configurations activées.

### 4. `analysis.py` (Analyse et pipeline)
Il s'agit du point d'entrée principal (script maître) qui orchestre l'ensemble de l'expérience.
- Il génère d'abord une instance complexe (ex: 100 capteurs, 30 zones).
- Il boucle ensuite pour exécuter l'heuristique et le solveur sur différents volumes de configurations initiales (ex: 10, 50, 100, 250, 500).
- Il utilise **Matplotlib** pour tracer et exporter un graphique (`analyse_duree_vie.png`). Ce graphique met en évidence l'évolution (et le plateau de convergence) de la durée de vie du réseau en fonction du nombre de configurations fournies au modèle linéaire.

## Prérequis

- **Python 3.x**
- **Matplotlib** (pour tracer les graphiques) : `pip install matplotlib`
- **Solveur GLPK** (`glpsol`) : Doit être installé sur votre système et être accessible globalement via votre variable d'environnement `PATH`. 
  - *Sous Windows, vous pouvez télécharger les binaires GLPK et ajouter le dossier contenant `glpsol.exe` à votre PATH.*

## Comment exécuter le projet ?

Pour lancer l'analyse comparative de bout en bout, exécutez la commande suivante à la racine du projet :

```bash
python analysis.py
```

Le script s'occupera d'afficher l'avancement dans le terminal. Il créera et supprimera à la volée les fichiers temporaires pour le solveur (`.lp` et `.txt`) et sauvegardera automatiquement le graphique de résultat dans votre dossier sous le nom de `analyse_duree_vie.png`.