import matplotlib.pyplot as plt
import os
from data_manager import SensorNetworkInstance
from heuristic import generate_elementary_configurations
from lp_solver import generate_lp_file, solve_lp, parse_solution

def run_analysis():
    num_zones = 30
    num_sensors = 100
    
    print("Génération de l'instance aléatoire...")
    instance = SensorNetworkInstance.generate_random(
        num_zones, num_sensors, min_lifetime=10, max_lifetime=100, coverage_prob=0.15
    )
    
    config_counts = [10, 50, 100, 250, 500]
    lifetimes = []
    
    print("Démarrage de l'analyse comparative...")
    for count in config_counts:
        configs = generate_elementary_configurations(instance, count)
        actual_count = len(configs)
        
        lp_file = f"prog_{actual_count}.lp"
        sol_file = f"sol_{actual_count}.txt"
        
        generate_lp_file(instance, configs, lp_file)
        success = solve_lp(lp_file, sol_file)
        
        if success:
            obj_val, active_vars = parse_solution(sol_file)
            lifetimes.append(obj_val)
            print(f"Configurations générées: {actual_count} | Durée de vie optimale: {obj_val:.2f} | Configs actives: {len(active_vars)}")
        else:
            lifetimes.append(0)
            print(f"Échec pour {actual_count} configurations.")
            
        if os.path.exists(lp_file): os.remove(lp_file)
        if os.path.exists(sol_file): os.remove(sol_file)
        
    plt.figure(figsize=(10, 6))
    plt.plot(config_counts, lifetimes, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=8)
    plt.title("Impact du nombre de configurations sur la durée de vie du réseau", fontsize=14)
    plt.xlabel("Nombre de configurations élémentaires (colonnes générées)", fontsize=12)
    plt.ylabel("Durée de vie maximale du réseau (Obj. LP)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig("analyse_duree_vie.png", dpi=300, bbox_inches='tight')
    print("Analyse terminée. Graphique sauvegardé sous 'analyse_duree_vie.png'.")

if __name__ == "__main__":
    run_analysis()
