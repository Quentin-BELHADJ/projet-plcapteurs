import subprocess
import os

def generate_lp_file(instance, configurations, filename="programme.lp"):
    with open(filename, 'w') as f:
        f.write("Maximize\n")
        obj_terms = [f"t_{j}" for j in range(len(configurations))]
        
        if not obj_terms:
            f.write(" obj: 0\n")
        else:
            f.write(" obj: " + " + ".join(obj_terms) + "\n")
            
        f.write("\nSubject To\n")
        for i in range(instance.num_sensors):
            terms = [f"t_{j}" for j, config in enumerate(configurations) if i in config]
            if terms:
                f.write(f" c_{i}: " + " + ".join(terms) + f" <= {instance.lifetimes[i]}\n")
                
        f.write("\nBounds\n")
        for j in range(len(configurations)):
            f.write(f" 0 <= t_{j}\n")
            
        f.write("\nEnd\n")

def solve_lp(lp_filename="programme.lp", sol_filename="solution.txt"):
    cmd = ["glpsol", "--cpxlp", lp_filename, "-o", sol_filename]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        print("Erreur: le solveur GLPK a retourné une erreur.")
        return False
    except FileNotFoundError:
        print("Erreur: glpsol introuvable. Veuillez vérifier qu'il est installé et dans le PATH.")
        return False

def parse_solution(sol_filename="solution.txt"):
    objective_value = 0.0
    variables = {}
    
    try:
        with open(sol_filename, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(lines):
            if line.startswith("Objective:"):
                parts = line.split()
                try:
                    idx = parts.index('=')
                    objective_value = float(parts[idx+1])
                except (ValueError, IndexError):
                    pass
                
            elif line.startswith("No. Column name"):
                var_start = i + 2
                for j in range(var_start, len(lines)):
                    if lines[j].strip() == "":
                        break
                    parts = lines[j].split()
                    if len(parts) >= 4:
                        var_name = parts[1]
                        try:
                            var_val = float(parts[3])
                            if var_val > 0:
                                variables[var_name] = var_val
                        except ValueError:
                            pass
        return objective_value, variables
    except Exception as e:
        print(f"Erreur lors du parsing: {e}")
        return 0.0, {}
