import random

def generate_elementary_configurations(instance, num_configs):
    configurations = set()
    attempts = 0
    max_attempts = num_configs * 50
    
    while len(configurations) < num_configs and attempts < max_attempts:
        attempts += 1
        uncovered_zones = set(range(instance.num_zones))
        current_config = []
        
        # 1. Génération d'une couverture aléatoire
        available_sensors = list(range(instance.num_sensors))
        random.shuffle(available_sensors)
        
        for sensor in available_sensors:
            covers_new = False
            for z in range(instance.num_zones):
                if instance.coverage_matrix[sensor][z] == 1 and z in uncovered_zones:
                    uncovered_zones.remove(z)
                    covers_new = True
            if covers_new:
                current_config.append(sensor)
            if not uncovered_zones:
                break
                
        if uncovered_zones:
            continue
            
        # 2. Réduction à une configuration élémentaire
        elementary_config = set(current_config)
        sensors_to_check = list(elementary_config)
        random.shuffle(sensors_to_check)
        
        for sensor in sensors_to_check:
            test_config = elementary_config - {sensor}
            covered_zones = set()
            for s in test_config:
                for z in range(instance.num_zones):
                    if instance.coverage_matrix[s][z] == 1:
                        covered_zones.add(z)
            
            if len(covered_zones) == instance.num_zones:
                elementary_config.remove(sensor)
                
        configurations.add(tuple(sorted(list(elementary_config))))
        
    return list(configurations)
