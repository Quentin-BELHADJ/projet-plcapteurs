#!/usr/bin/env python3
"""
Script principal — Ordonnancement adaptatif de capteurs de surveillance.

Ce script orchestre la résolution complète du problème :
1. Chargement/création d'une instance
2. Génération de configurations élémentaires
3. Écriture et résolution du programme linéaire
4. Affichage de la solution optimale
"""

import sys
import os
from data_manager import SensorNetworkInstance
from heuristic import generate_elementary_configurations, display_configurations
from lp_solver import solve, display_solution, generate_lp_file


def select_data_source():
    """Menu de sélection de la source de données."""
    print("\n" + "=" * 60)
    print("  ORDONNANCEMENT ADAPTATIF DE CAPTEURS DE SURVEILLANCE")
    print("=" * 60)
    print("\n  Source des données :\n")
    print("    1. Charger depuis un fichier")
    print("    2. Générer une instance aléatoire")
    print("    3. Saisie au clavier")
    print()

    choice = input("  Votre choix (1/2/3) : ").strip()

    if choice == "1":
        filepath = input("  Chemin du fichier : ").strip()
        if not os.path.exists(filepath):
            print(f"  Erreur : le fichier '{filepath}' n'existe pas.")
            sys.exit(1)
        instance = SensorNetworkInstance.from_file(filepath)
        print(f"\n  Instance chargée depuis '{filepath}'.")

    elif choice == "2":
        n_sensors = int(input("  Nombre de capteurs (N) : "))
        n_zones = int(input("  Nombre de zones (M) : "))
        instance = SensorNetworkInstance.generate_random(n_zones, n_sensors)
        print(f"\n  Instance aléatoire générée ({n_sensors} capteurs, {n_zones} zones).")

    elif choice == "3":
        instance = SensorNetworkInstance.from_keyboard()

    else:
        print("  Choix invalide.")
        sys.exit(1)

    return instance


def select_heuristic():
    """Menu de sélection de l'heuristique."""
    print("  Heuristique de génération des configurations :\n")
    print("    1. Gloutonne aléatoire (diversité maximale)")
    print("    2. Greedy couverture maximale (configs compactes)")
    print()

    choice = input("  Votre choix (1/2) : ").strip()
    if choice == "2":
        return "greedy"
    return "random"


def main():
    """Point d'entrée principal."""

    # --- Gestion des arguments en ligne de commande ---
    if len(sys.argv) > 1:
        # Mode non-interactif : python main.py <fichier> [num_configs] [method]
        filepath = sys.argv[1]
        num_configs = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        method = sys.argv[3] if len(sys.argv) > 3 else "random"

        if not os.path.exists(filepath):
            print(f"Erreur : le fichier '{filepath}' n'existe pas.")
            sys.exit(1)

        instance = SensorNetworkInstance.from_file(filepath)
        print(f"\n  Instance chargée depuis '{filepath}'.")
    else:
        # Mode interactif
        instance = select_data_source()

        method = select_heuristic()

        num_configs_str = input("  Nombre de configurations à générer : ").strip()
        num_configs = int(num_configs_str) if num_configs_str else 50

    # --- Affichage de l'instance ---
    instance.display()

    # --- Génération des configurations élémentaires ---
    print(f"  Génération des configurations élémentaires (méthode: {method})...")
    configurations = generate_elementary_configurations(instance, num_configs, method=method)

    if not configurations:
        print("  Erreur : aucune configuration élémentaire n'a pu être générée.")
        sys.exit(1)

    display_configurations(instance, configurations)

    # --- Écriture et résolution du programme linéaire ---
    lp_file = "programme.lp"
    sol_file = "solution.txt"

    print("  Résolution du programme linéaire...")
    objective_value, variables, solver_used = solve(
        instance, configurations, lp_file, sol_file
    )

    print(f"  Solveur utilisé : {solver_used}")

    if objective_value > 0:
        display_solution(instance, configurations, objective_value, variables)
        print(f"  Fichier LP généré : {lp_file}")
    else:
        print("  Échec de la résolution du programme linéaire.")


if __name__ == "__main__":
    main()
