import subprocess
import os
import numpy as np
from scipy.optimize import linprog


def generate_lp_file(instance, configurations, filename="programme.lp"):
    """Écrit le programme linéaire au format CPLEX (.lp) pour GLPK.

    Le PL maximise la somme des temps d'activation des configurations
    sous contrainte de durée de vie de chaque capteur.
    """
    with open(filename, 'w') as f:
        # Fonction objectif
        f.write("Maximize\n")
        obj_terms = [f"t{j}" for j in range(len(configurations))]
        if not obj_terms:
            f.write(" obj: 0\n")
        else:
            f.write(" obj: " + " + ".join(obj_terms) + "\n")

        # Contraintes de durée de vie des capteurs
        f.write("\nSubject To\n")
        for i in range(instance.num_sensors):
            terms = [f"t{j}" for j, config in enumerate(configurations) if i in config]
            if terms:
                f.write(f" c{i}: " + " + ".join(terms) + f" <= {instance.lifetimes[i]:g}\n")

        # Bornes (toutes les variables >= 0)
        f.write("\nBounds\n")
        for j in range(len(configurations)):
            f.write(f" 0 <= t{j}\n")

        f.write("\nEnd\n")

    return filename


def solve_with_glpk(lp_filename="programme.lp", sol_filename="solution.txt"):
    """Résout le PL avec GLPK (glpsol) via la ligne de commande.

    Returns:
        True si la résolution a réussi, False sinon.
    """
    cmd = ["glpsol", "--cpxlp", lp_filename, "-o", sol_filename]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print("  [GLPK] Erreur : le solveur a retourné une erreur.")
        return False
    except FileNotFoundError:
        return False


def parse_glpk_solution(sol_filename="solution.txt"):
    """Parse le fichier de solution GLPK.

    Returns:
        (objective_value, dict{var_name: value})
    """
    objective_value = 0.0
    variables = {}

    try:
        with open(sol_filename, 'r') as f:
            lines = f.readlines()

        in_columns = False
        for i, line in enumerate(lines):
            # Chercher la valeur de la fonction objectif
            if "Objective:" in line or "obj =" in line:
                parts = line.split()
                for k, p in enumerate(parts):
                    if p == '=':
                        try:
                            objective_value = float(parts[k + 1])
                        except (ValueError, IndexError):
                            pass
                        break

            # Détecter le début du tableau des variables
            if "Column name" in line:
                in_columns = True
                continue

            if in_columns:
                stripped = line.strip()
                if stripped == "" or stripped.startswith("---"):
                    if variables:  # On a déjà lu des variables, fin du tableau
                        in_columns = False
                    continue

                parts = stripped.split()
                if len(parts) >= 4:
                    try:
                        var_name = parts[1]
                        # La valeur d'activité est en position 3 (après no, name, status)
                        var_val = float(parts[3])
                        if var_val > 1e-10:
                            variables[var_name] = var_val
                    except (ValueError, IndexError):
                        pass

        return objective_value, variables
    except Exception as e:
        print(f"  Erreur lors du parsing GLPK: {e}")
        return 0.0, {}


def solve_with_scipy(instance, configurations):
    """Résout le PL directement avec scipy.optimize.linprog (méthode du simplexe révisé).

    Le problème est formulé comme :
        max  sum(t_j)
        s.c. pour chaque capteur i : sum(t_j pour j où i ∈ config_j) <= lifetime_i
             t_j >= 0

    linprog minimise, donc on minimise -sum(t_j).

    Returns:
        (objective_value, dict{var_name: value}) ou (0.0, {}) en cas d'échec.
    """
    n_configs = len(configurations)
    if n_configs == 0:
        return 0.0, {}

    # Coefficients de la fonction objectif (minimiser -1 * sum)
    c = [-1.0] * n_configs

    # Construire la matrice des contraintes A_ub et le vecteur b_ub
    # Pour chaque capteur, on a : sum(t_j pour j où capteur i ∈ config_j) <= lifetime_i
    A_ub = []
    b_ub = []

    for i in range(instance.num_sensors):
        row = [0.0] * n_configs
        has_constraint = False
        for j, config in enumerate(configurations):
            if i in config:
                row[j] = 1.0
                has_constraint = True
        if has_constraint:
            A_ub.append(row)
            b_ub.append(instance.lifetimes[i])

    if not A_ub:
        return 0.0, {}

    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)

    # Bornes : t_j >= 0 (par défaut dans linprog)
    bounds = [(0, None)] * n_configs

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if result.success:
        objective_value = -result.fun  # Inversion car on a minimisé -f
        variables = {}
        for j in range(n_configs):
            if result.x[j] > 1e-10:
                variables[f"t{j}"] = result.x[j]
        return objective_value, variables
    else:
        print(f"  [scipy] Échec de la résolution : {result.message}")
        return 0.0, {}


def solve(instance, configurations, lp_filename="programme.lp", sol_filename="solution.txt"):
    """Fonction unifiée de résolution.

    1. Génère toujours le fichier .lp (format CPLEX) comme demandé par le sujet
    2. Essaie GLPK en priorité
    3. Si GLPK indisponible, utilise scipy comme fallback

    Returns:
        (objective_value, dict{var_name: value}, solver_used)
    """
    # Toujours générer le fichier .lp
    generate_lp_file(instance, configurations, lp_filename)

    # Essayer GLPK d'abord
    if solve_with_glpk(lp_filename, sol_filename):
        obj, variables = parse_glpk_solution(sol_filename)
        return obj, variables, "GLPK"

    # Fallback vers scipy
    obj, variables = solve_with_scipy(instance, configurations)
    return obj, variables, "scipy"


def display_solution(instance, configurations, objective_value, variables):
    """Affiche la solution optimale de manière détaillée."""
    print(f"\n{'='*60}")
    print(f"  SOLUTION OPTIMALE")
    print(f"{'='*60}")
    print(f"\n  Durée de vie du réseau : {objective_value:.4f} unités de temps\n")

    # Afficher les configurations actives
    print(f"  {'Configuration':<20} {'Capteurs':<25} {'Temps (t*)':<12}")
    print(f"  {'-'*20} {'-'*25} {'-'*12}")

    for j, config in enumerate(configurations):
        var_name = f"t{j}"
        t_val = variables.get(var_name, 0.0)
        if t_val > 1e-10:
            sensors_str = ", ".join(f"s{s+1}" for s in config)
            print(f"  u{j+1:<18} ({sensors_str}){' '*(23-len(sensors_str))} {t_val:>10.4f}")

    # Bilan par capteur
    print(f"\n  {'Capteur':<10} {'Temps actif':<15} {'Durée de vie':<15} {'Restant':<15} {'Statut'}")
    print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")

    for i in range(instance.num_sensors):
        active_time = 0.0
        for j, config in enumerate(configurations):
            if i in config:
                var_name = f"t{j}"
                active_time += variables.get(var_name, 0.0)

        remaining = instance.lifetimes[i] - active_time
        if active_time < 1e-10:
            status = "⏸  En veille"
        elif remaining < 1e-6:
            status = "🔋 Épuisé"
        else:
            status = "✅ Énergie restante"

        print(f"  s{i+1:<8} {active_time:>13.4f} {instance.lifetimes[i]:>13.1f} {remaining:>13.4f}   {status}")
    print()

def solve_with_duals(instance, configurations):
    """Résout le LP et retourne les prix duaux par capteur.
    
    Utilisé par DualGuidedGenerator et ColumnGeneration pour identifier
    les capteurs goulots d'étranglement (contrainte binding ↔ dual > 0).

    Returns:
        (obj, variables, sensor_duals)
        sensor_duals : dict { sensor_idx: dual_price }
    """
    n_configs = len(configurations)
    if n_configs == 0:
        return 0.0, {}, {}

    c = [-1.0] * n_configs
    A_ub, b_ub = [], []
    active_sensors = []  # capteurs qui apparaissent dans au moins une config

    for i in range(instance.num_sensors):
        row = [1.0 if i in config else 0.0 for config in configurations]
        if any(r > 0 for r in row):
            A_ub.append(row)
            b_ub.append(instance.lifetimes[i])
            active_sensors.append(i)

    if not A_ub:
        return 0.0, {}, {}

    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                     bounds=[(0, None)] * n_configs, method='highs',
                     options={'disp': False})

    if not result.success:
        return 0.0, {}, {}

    obj = -result.fun
    variables = {f"t{j}": result.x[j] for j in range(n_configs) if result.x[j] > 1e-10}

    # Prix duaux : scipy/HiGHS les retourne dans ineqlin.marginals
    # Convention : marginal négatif pour une contrainte <= → on inverse
    sensor_duals = {}
    if hasattr(result, 'ineqlin') and result.ineqlin is not None:
        marginals = result.ineqlin.marginals
        for idx, i in enumerate(active_sensors):
            if idx < len(marginals):
                sensor_duals[i] = float(-marginals[idx])  # positif = contrainte binding

    return obj, variables, sensor_duals