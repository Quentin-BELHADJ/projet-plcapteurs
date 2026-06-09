#!/usr/bin/env python3
"""
Analyse et expérimentation — Parties 4 et 5 du projet.

Partie 4 : Résolution des différentes instances de problème.
Partie 5 : Analyse de l'influence du nombre et du type de configurations
            sur la durée de vie du réseau.
"""

import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif pour serveurs sans display

from data_manager import SensorNetworkInstance
from heuristic import generate_elementary_configurations
from lp_solver import solve


# =====================================================================
#  Partie 4 : Expérimentation sur les instances fournies
# =====================================================================

def run_all_instances():
    """Résout toutes les instances de test disponibles et affiche un tableau de résultats."""
    test_files = [
        ("fichier-exemple.txt", "Exemple du sujet"),
        ("moyen_test_2.txt",    "Moyen (20 capteurs, 10 zones)"),
        ("moyen_test_3.txt",    "Moyen (10 capteurs, 10 zones)"),
        ("gros_test_1.txt",     "Gros (100 capteurs, 200 zones)"),
        ("maxi_test_1.txt",     "Maxi (1000 capteurs, 500 zones)"),
    ]

    num_configs = 1000  # Nombre de configs à générer par défaut

    print("\n" + "=" * 90)
    print("  PARTIE 4 — Résolution des instances de test")
    print("=" * 90)

    results = []

    print(f"\n  {'Fichier':<22} {'N':>5} {'M':>5} {'Configs':>8} {'Durée de vie':>14} {'Solveur':<8}")
    print(f"  {'-'*22} {'-'*5} {'-'*5} {'-'*8} {'-'*14} {'-'*8}")

    for filename, description in test_files:
        if not os.path.exists(filename):
            print(f"  {filename:<22} {'—':>5} {'—':>5} {'—':>8} {'FICHIER ABSENT':>14} {'—':<8}")
            continue

        instance = SensorNetworkInstance.from_file(filename)

        # Adapter le nombre de configs à la taille du problème
        adapted_configs = min(num_configs, instance.num_sensors * 5)

        configs = generate_elementary_configurations(instance, adapted_configs, method="random")
        actual_count = len(configs)

        obj_val, variables, solver = solve(
            instance, configs,
            lp_filename=f"prog_{filename}",
            sol_filename=f"sol_{filename}"
        )

        results.append({
            "filename": filename,
            "description": description,
            "N": instance.num_sensors,
            "M": instance.num_zones,
            "configs": actual_count,
            "objective": obj_val,
            "solver": solver,
            "active": len(variables),
        })

        print(f"  {filename:<22} {instance.num_sensors:>5} {instance.num_zones:>5} "
              f"{actual_count:>8} {obj_val:>14.4f} {solver:<8}")

        # Nettoyage des fichiers temporaires
        for tmp in [f"prog_{filename}", f"sol_{filename}"]:
            if os.path.exists(tmp):
                os.remove(tmp)

    print()
    return results


# =====================================================================
#  Partie 5 : Analyse de l'influence des configurations
# =====================================================================

def analyze_num_configs(instance, instance_name="instance"):
    """Analyse l'influence du NOMBRE de configurations sur la durée de vie."""
    config_counts = [5, 10, 25, 50, 100, 200, 500, 1000, 5000]
    lifetimes_random = []
    lifetimes_greedy = []
    actual_counts_random = []
    actual_counts_greedy = []

    print(f"\n  Analyse : influence du nombre de configurations ({instance_name})")
    print(f"  {'Demandé':>10} {'Random (n)':>12} {'Random (obj)':>14} {'Greedy (n)':>12} {'Greedy (obj)':>14}")
    print(f"  {'-'*10} {'-'*12} {'-'*14} {'-'*12} {'-'*14}")

    for count in config_counts:
        # Méthode random
        configs_r = generate_elementary_configurations(instance, count, method="random")
        obj_r, _, _ = solve(instance, configs_r, f"tmp_r.lp", f"tmp_r.txt")
        lifetimes_random.append(obj_r)
        actual_counts_random.append(len(configs_r))

        # Méthode greedy
        configs_g = generate_elementary_configurations(instance, count, method="greedy")
        obj_g, _, _ = solve(instance, configs_g, f"tmp_g.lp", f"tmp_g.txt")
        lifetimes_greedy.append(obj_g)
        actual_counts_greedy.append(len(configs_g))

        print(f"  {count:>10} {len(configs_r):>12} {obj_r:>14.4f} {len(configs_g):>12} {obj_g:>14.4f}")

    # Nettoyage
    for tmp in ["tmp_r.lp", "tmp_r.txt", "tmp_g.lp", "tmp_g.txt"]:
        if os.path.exists(tmp):
            os.remove(tmp)

    return config_counts, lifetimes_random, lifetimes_greedy, actual_counts_random, actual_counts_greedy


def generate_plots(config_counts, lifetimes_random, lifetimes_greedy,
                   actual_random, actual_greedy, instance_name="instance"):
    """Génère les graphiques d'analyse."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Graphique 1 : Durée de vie vs nombre de configurations demandées ---
    ax1 = axes[0]
    ax1.plot(config_counts, lifetimes_random, marker='o', linestyle='-',
             color='#2196F3', linewidth=2, markersize=7, label='Aléatoire glouton')
    ax1.plot(config_counts, lifetimes_greedy, marker='s', linestyle='--',
             color='#FF5722', linewidth=2, markersize=7, label='Greedy max couverture')
    ax1.set_title("Influence du nombre de configurations\nsur la durée de vie du réseau",
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel("Nombre de configurations demandées", fontsize=11)
    ax1.set_ylabel("Durée de vie optimale (unités de temps)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Graphique 2 : Comparaison des deux heuristiques (configs réellement générées) ---
    ax2 = axes[1]
    ax2.plot(actual_random, lifetimes_random, marker='o', linestyle='-',
             color='#2196F3', linewidth=2, markersize=7, label='Aléatoire glouton')
    ax2.plot(actual_greedy, lifetimes_greedy, marker='s', linestyle='--',
             color='#FF5722', linewidth=2, markersize=7, label='Greedy max couverture')
    ax2.set_title("Durée de vie vs configurations\neffectivement générées",
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel("Nombre de configurations effectivement générées", fontsize=11)
    ax2.set_ylabel("Durée de vie optimale (unités de temps)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    output_file = f"analyse_{instance_name}.png"
    plt.savefig(output_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Graphique sauvegardé : {output_file}")
    return output_file


def run_analysis():
    """Lance l'analyse complète (Parties 4 et 5)."""

    # ---- Partie 4 : Résolution des instances ----
    results = run_all_instances()

    # ---- Partie 5 : Analyse sur une instance de taille moyenne ----
    print("\n" + "=" * 90)
    print("  PARTIE 5 — Analyse de l'influence des configurations")
    print("=" * 90)

    # Utiliser gros_test_1 pour l'analyse (10 capteurs, 10 zones — rapide)
    analysis_file = "gros_test_1.txt"
    if os.path.exists(analysis_file):
        instance = SensorNetworkInstance.from_file(analysis_file)
        instance_name = "gros_test_1"
    else:
        # Fallback : générer une instance aléatoire
        instance = SensorNetworkInstance.generate_random(15, 30, min_lifetime=10, max_lifetime=100, coverage_prob=0.2)
        instance_name = "aleatoire_30x15"

    print(f"\n  Instance utilisée : {instance_name} ({instance.num_sensors} capteurs, {instance.num_zones} zones)")

    config_counts, lt_random, lt_greedy, n_random, n_greedy = analyze_num_configs(instance, instance_name)
    generate_plots(config_counts, lt_random, lt_greedy, n_random, n_greedy, instance_name)

    # Analyse sur une instance plus grande si disponible
    analysis_file_2 = "moyen_test_2.txt"
    if os.path.exists(analysis_file_2):
        instance2 = SensorNetworkInstance.from_file(analysis_file_2)
        instance_name2 = "moyen_test_2"
        print(f"\n  Instance utilisée : {instance_name2} ({instance2.num_sensors} capteurs, {instance2.num_zones} zones)")
        cc2, lr2, lg2, nr2, ng2 = analyze_num_configs(instance2, instance_name2)
        generate_plots(cc2, lr2, lg2, nr2, ng2, instance_name2)

    print("\n" + "=" * 90)
    print("  Analyse terminée.")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_analysis()
