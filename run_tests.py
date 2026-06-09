#!/usr/bin/env python3
"""
Lance la résolution de toutes les instances de test d'un coup.

Usage:
    python run_tests.py                     # méthode random, 500 configs
    python run_tests.py greedy              # méthode greedy
    python run_tests.py random 2000         # méthode random, 2000 configs
"""

import sys
import os
import time
from data_manager import SensorNetworkInstance
from heuristic import generate_elementary_configurations
from lp_solver import solve, display_solution

# --- Arguments ---
method = sys.argv[1] if len(sys.argv) > 1 else "random"
num_configs = int(sys.argv[2]) if len(sys.argv) > 2 else 500

# --- Instances de test ---
TEST_FILES = [
    "fichier-exemple.txt",
    "moyen_test_2.txt",
    "moyen_test_3.txt",
    "gros_test_1.txt",
    "maxi_test_1.txt",
]

PROF_RESULTS = [8.5, 104, 395, 1772, 13645]

print(f"\n{'='*80}")
print(f"  RÉSOLUTION DE TOUTES LES INSTANCES")
print(f"  Méthode : {method}  |  Configs demandées : {num_configs}")
print(f"{'='*80}\n")

header = f"  {'#':<4} {'Fichier':<22} {'N':>6} {'M':>6} {'Configs':>8} {'Durée vie':>12} {'Prof':>10} {'Écart':>10} {'Temps':>8}"
print(header)
print(f"  {'─'*4} {'─'*22} {'─'*6} {'─'*6} {'─'*8} {'─'*12} {'─'*10} {'─'*10} {'─'*8}")

total_ok = 0

for idx, filename in enumerate(TEST_FILES):
    if not os.path.exists(filename):
        print(f"  {idx+1:<4} {filename:<22} {'FICHIER ABSENT':>50}")
        continue

    t0 = time.time()

    instance = SensorNetworkInstance.from_file(filename)
    configs = generate_elementary_configurations(instance, num_configs, method=method)
    obj, variables, solver = solve(instance, configs, f"_tmp_{idx}.lp", f"_tmp_{idx}.txt")

    elapsed = time.time() - t0

    prof = PROF_RESULTS[idx]
    ecart = obj - prof
    sign = "+" if ecart >= 0 else ""
    status = "✅" if ecart >= -0.01 else "❌"

    print(f"  {idx+1:<4} {filename:<22} {instance.num_sensors:>6} {instance.num_zones:>6} "
          f"{len(configs):>8} {obj:>12.2f} {prof:>10.1f} {sign}{ecart:>9.1f} {elapsed:>7.1f}s {status}")

    if ecart >= -0.01:
        total_ok += 1

    # Nettoyage
    for tmp in [f"_tmp_{idx}.lp", f"_tmp_{idx}.txt"]:
        if os.path.exists(tmp):
            os.remove(tmp)

print(f"\n  Résultat : {total_ok}/{len(TEST_FILES)} instances ≥ prof")
print(f"{'='*80}\n")
