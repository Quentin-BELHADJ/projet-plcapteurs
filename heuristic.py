import random


def _is_valid_configuration(instance, config):
    """Vérifie qu'une configuration couvre toutes les zones."""
    covered = set()
    for s in config:
        covered.update(instance.get_zones_covered_by(s))
    return len(covered) == instance.num_zones


def _is_elementary(instance, config):
    """Vérifie qu'une configuration est élémentaire (minimale).
    Retirer n'importe quel capteur doit briser la couverture complète."""
    for s in config:
        reduced = config - {s}
        if _is_valid_configuration(instance, reduced):
            return False
    return True


def _reduce_to_elementary(instance, config):
    """Réduit une configuration valide à une configuration élémentaire
    en retirant les capteurs superflus."""
    elementary = set(config)
    sensors_to_check = list(elementary)
    random.shuffle(sensors_to_check)

    for sensor in sensors_to_check:
        test_config = elementary - {sensor}
        if _is_valid_configuration(instance, test_config):
            elementary.remove(sensor)

    return frozenset(elementary)


def generate_random_greedy(instance, num_configs):
    """Heuristique 1 : Génération aléatoire gloutonne.

    À chaque itération :
    1. Mélange aléatoirement les capteurs
    2. Ajoute greedily ceux qui couvrent de nouvelles zones
    3. Réduit la configuration obtenue pour la rendre élémentaire

    Cette approche maximise la diversité des configurations générées.
    """
    configurations = set()
    attempts = 0
    max_attempts = num_configs * 100

    while len(configurations) < num_configs and attempts < max_attempts:
        attempts += 1
        uncovered_zones = set(range(instance.num_zones))
        current_config = []

        available_sensors = list(range(instance.num_sensors))
        random.shuffle(available_sensors)

        for sensor in available_sensors:
            new_zones = instance.get_zones_covered_by(sensor) & uncovered_zones
            if new_zones:
                uncovered_zones -= new_zones
                current_config.append(sensor)
            if not uncovered_zones:
                break

        if uncovered_zones:
            continue

        # Réduction à une configuration élémentaire
        elementary = _reduce_to_elementary(instance, set(current_config))
        configurations.add(elementary)

    return [tuple(sorted(c)) for c in configurations]


def generate_greedy_max_coverage(instance, num_configs):
    """Heuristique 2 : Greedy déterministe par couverture maximale.

    À chaque itération :
    1. Choisit le capteur couvrant le plus de zones non couvertes
       (avec bruit aléatoire pour la diversité)
    2. Réduit la configuration obtenue pour la rendre élémentaire

    Cette approche tend à produire des configurations avec moins de capteurs.
    """
    configurations = set()
    attempts = 0
    max_attempts = num_configs * 100

    while len(configurations) < num_configs and attempts < max_attempts:
        attempts += 1
        uncovered_zones = set(range(instance.num_zones))
        current_config = []
        used_sensors = set()

        while uncovered_zones:
            best_sensor = None
            best_score = -1

            for s in range(instance.num_sensors):
                if s in used_sensors:
                    continue
                new_zones = instance.get_zones_covered_by(s) & uncovered_zones
                # Score = nombre de nouvelles zones + petit bruit pour la diversité
                score = len(new_zones) + random.uniform(0, 0.5)
                if score > best_score and len(new_zones) > 0:
                    best_score = score
                    best_sensor = s

            if best_sensor is None:
                break

            uncovered_zones -= instance.get_zones_covered_by(best_sensor)
            current_config.append(best_sensor)
            used_sensors.add(best_sensor)

        if uncovered_zones:
            continue

        elementary = _reduce_to_elementary(instance, set(current_config))
        configurations.add(elementary)

    return [tuple(sorted(c)) for c in configurations]


def generate_elementary_configurations(instance, num_configs, method="random"):
    """Fonction wrapper pour la génération de configurations élémentaires.

    Args:
        instance: Instance du problème (SensorNetworkInstance).
        num_configs: Nombre de configurations souhaitées.
        method: "random" (heuristique gloutonne aléatoire)
                ou "greedy" (heuristique greedy déterministe).

    Returns:
        Liste de tuples, chaque tuple étant une configuration élémentaire
        (indices de capteurs triés, 0-based).
    """
    if method == "greedy":
        return generate_greedy_max_coverage(instance, num_configs)
    else:
        return generate_random_greedy(instance, num_configs)


def display_configurations(instance, configurations):
    """Affiche les configurations élémentaires de manière lisible."""
    print(f"\n  {len(configurations)} configurations élémentaires générées :\n")
    for idx, config in enumerate(configurations):
        sensors_str = ", ".join(f"s{s+1}" for s in config)
        # Calculer les zones couvertes pour vérification
        covered = set()
        for s in config:
            covered.update(instance.get_zones_covered_by(s))
        zones_str = ", ".join(f"z{z+1}" for z in sorted(covered))
        print(f"    u{idx+1} = ({sensors_str})")
    print()
