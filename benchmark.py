#!/usr/bin/env python3
"""
benchmark.py — Comparaison des méthodes de génération de configurations.

Compare les 5 méthodes sur les instances de test :
  1. random_greedy     (heuristique 1 originale)
  2. greedy_hybrid     (heuristique 2 originale)
  3. random_pruning    (nouvelle — scoring dynamique + pruning probabiliste)
  4. dual_guided       (nouvelle — guidé par les prix duaux LP)
  5. column_generation (nouvelle — génération de colonnes, méthode exacte)

Usage :
  python benchmark.py                         # toutes les méthodes
  python benchmark.py random_greedy           # une méthode
  python benchmark.py dual_guided 200         # méthode + nb configs
  python benchmark.py all 500                 # toutes + nb configs
"""

import sys
import time
import os
import csv
from datetime import datetime
from data_manager import SensorNetworkInstance
from heuristic import generate_elementary_configurations
from heuristic_advanced import (
    generate_with_random_pruning,
    generate_with_dual_guidance,
    generate_with_column_generation,
    generate_with_simulated_annealing,
    generate_with_real_time_greedy,
)
from lp_solver import solve

# Résultats de référence du professeur
PROF_RESULTS = {
    "fichier-exemple.txt": 8.5,
    "moyen_test_2.txt": 104.0,
    "moyen_test_3.txt": 395.0,
    "gros_test_1.txt": 1772,
    "maxi_test_1.txt": 13645,
}

TEST_FILES = [
    ("fichier-exemple.txt",  "Exemple (4c, 3z)"),
    ("moyen_test_2.txt",     "Moyen-2 (20c, 10z)"),
    ("moyen_test_3.txt",     "Moyen-3 (10c, 10z)"),
    ("gros_test_1.txt",      "Gros-1 (100c, 200z)"),
    ("maxi_test_1.txt",      "Maxi-1 (1000c, 500z)")
]

NUM_CONFIGS = 100  # Limite pour rester rapide


def run_method(instance, method_name, num_configs):
    t0 = time.time()
    configs = []
    cg_history = None

    if method_name == "random_greedy":
        configs = generate_elementary_configurations(instance, num_configs, method="random")
    elif method_name == "greedy_hybrid":
        configs = generate_elementary_configurations(instance, num_configs, method="greedy")
    elif method_name == "random_pruning":
        configs = generate_with_random_pruning(instance, num_configs,
                                                alpha=1.5, beta=0.5,
                                                noise=0.3, p_prune=0.4)
    elif method_name == "dual_guided":
        configs = generate_with_dual_guidance(instance, num_configs,
                                               dual_weight=2.0, iterations=5)
    elif method_name == "column_generation":
        configs, cg_history, _ = generate_with_column_generation(
            instance, num_configs, pricing_mode="auto"
        )
    elif method_name == "simulated_annealing":
        configs = generate_with_simulated_annealing(instance, num_configs)
    elif method_name == "real_time_greedy":
        configs = generate_with_real_time_greedy(instance, num_configs)

    gen_time = time.time() - t0

    if not configs:
        return None, 0, 0.0, gen_time, cg_history

    t1 = time.time()
    obj, _, solver = solve(instance, configs, "_tmp_bm.lp", "_tmp_bm.txt")
    solve_time = time.time() - t1

    for f in ["_tmp_bm.lp", "_tmp_bm.txt"]:
        if os.path.exists(f):
            os.remove(f)

    return obj, len(configs), gen_time + solve_time, gen_time, cg_history


ALL_METHODS = [
    ("random_greedy",     "Random Greedy"),
    ("greedy_hybrid",     "Greedy Hybride"),
    ("random_pruning",    "Random Pruning"),
    ("column_generation", "Column Generation"),
    ("simulated_annealing", "Simulated Annealing"),
    ("real_time_greedy",  "Real-Time Greedy"),
]

VALID_IDS = {m[0] for m in ALL_METHODS}


def parse_args():
    """Parse les arguments CLI.

    Returns:
        (methods_to_run, num_configs)
    """
    args = sys.argv[1:]

    # Extraire num_configs si le dernier argument est un entier
    num_configs = NUM_CONFIGS
    if args and args[-1].isdigit():
        num_configs = int(args[-1])
        args = args[:-1]

    # Extraire les méthodes
    if not args or args[0] in ("all", ""):
        methods = ALL_METHODS
    else:
        unknown = [a for a in args if a not in VALID_IDS]
        if unknown:
            print(f"\n  Méthode(s) inconnue(s) : {', '.join(unknown)}")
            print(f"  Méthodes disponibles   : {', '.join(VALID_IDS)}")
            print(f"  Usage : python benchmark_advanced.py [méthode ...] [num_configs]\n")
            sys.exit(1)
        methods = [(mid, label) for mid, label in ALL_METHODS if mid in args]

    return methods, num_configs


def main():
    methods, num_configs = parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    algo_names = "_".join(m[0] for m in methods)
    if len(algo_names) > 30:
        algo_names = "multiple_algos"
    csv_filename = f"results_{timestamp}_{algo_names}_{num_configs}configs.csv"
    csv_data = []

    print(f"\n{'='*90}")
    print(f"  BENCHMARK DES MÉTHODES DE GÉNÉRATION  (num_configs={num_configs})")
    if len(methods) < len(ALL_METHODS):
        print(f"  Méthodes sélectionnées : {', '.join(m[0] for m in methods)}")
    print(f"{'='*90}")

    for filename, desc in TEST_FILES:
        if not os.path.exists(filename):
            print(f"  ⚠ Fichier absent : {filename}")
            continue

        instance = SensorNetworkInstance.from_file(filename)
        prof = PROF_RESULTS.get(filename, None)

        print(f"\n  ┌─ {desc}  ({filename})")
        if prof:
            print(f"  │  Référence prof : {prof:.1f}")
        print(f"  │")
        print(f"  │  {'Méthode':<22} {'Configs':>7} {'Obj':>10} {'vs Prof':>9} {'Temps':>7}")
        print(f"  │  {'─'*22} {'─'*7} {'─'*10} {'─'*9} {'─'*7}")

        best_obj = 0.0
        results = []

        for method_id, method_label in methods:
            obj, n_configs, total_time, gen_time, cg_history = run_method(
                instance, method_id, num_configs
            )

            if obj is None:
                print(f"  │  {method_label:<22} {'—':>7} {'ÉCHEC':>10}")
                continue

            vs_prof = f"{obj - prof:+.2f}" if prof else "—"
            marker = "◄ BEST" if obj > best_obj else ""
            best_obj = max(best_obj, obj)

            print(f"  │  {method_label:<22} {n_configs:>7} {obj:>10.4f} {vs_prof:>9} {total_time:>6.2f}s  {marker}")
            results.append((method_label, obj, n_configs, total_time))

            csv_data.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Heure": datetime.now().strftime("%H:%M:%S"),
                "Instance": filename,
                "Algorithme": method_id,
                "Nb_Solutions_Explorees": n_configs,
                "Score_Objectif": obj,
                "Temps_Execution_s": round(total_time, 2)
            })

            # Afficher la convergence de la génération de colonnes
            if cg_history and len(cg_history) > 1:
                print(f"  │    └─ CG : {len(cg_history)} itérations, "
                      f"obj initial={cg_history[0][1]:.2f} → final≈{cg_history[-1][1]:.2f}")

        print(f"  └─ Meilleur : {best_obj:.4f}")

    if csv_data:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
            writer.writeheader()
            writer.writerows(csv_data)
        print(f"\n  [+] Fichier de résultats généré : {csv_filename}")

    print(f"\n{'='*90}")
    print(f"  Benchmark terminé.")
    print(f"{'='*90}\n")


if __name__ == "__main__":
    main()
