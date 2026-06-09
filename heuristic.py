import random
from itertools import combinations


# ═══════════════════════════════════════════════════════════════════
#  Fonctions utilitaires
# ═══════════════════════════════════════════════════════════════════

def _is_valid_configuration(instance, config):
    """Vérifie qu'une configuration couvre toutes les zones."""
    covered = set()
    for s in config:
        covered.update(instance.get_zones_covered_by(s))
    return len(covered) == instance.num_zones


def _is_elementary(instance, config):
    """Vérifie qu'une configuration est élémentaire (minimale) :
    retirer n'importe quel capteur doit briser la couverture complète."""
    config = list(config)
    for i, s in enumerate(config):
        if _is_valid_configuration(instance, config[:i] + config[i + 1:]):
            return False
    return True


def _reduce_to_elementary(instance, config):
    """Réduit une configuration valide à une configuration élémentaire
    en retirant les capteurs superflus dans un ordre aléatoire."""
    elementary = set(config)
    sensors_to_check = list(elementary)
    random.shuffle(sensors_to_check)
    for sensor in sensors_to_check:
        reduced = elementary - {sensor}
        if _is_valid_configuration(instance, reduced):
            elementary = reduced
    return frozenset(elementary)


# ═══════════════════════════════════════════════════════════════════
#  Heuristique 1 : Gloutonne aléatoire (conservée)
# ═══════════════════════════════════════════════════════════════════

def generate_random_greedy(instance, num_configs):
    """Heuristique 1 : Génération aléatoire gloutonne.

    À chaque itération :
    1. Mélange aléatoirement les capteurs
    2. Ajoute greedilement ceux qui couvrent de nouvelles zones
    3. Réduit la configuration obtenue pour la rendre élémentaire

    Conservée comme fallback et pour la comparaison en Partie 5.
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

        elementary = _reduce_to_elementary(instance, set(current_config))
        configurations.add(elementary)

    return [tuple(sorted(c)) for c in configurations]


# ═══════════════════════════════════════════════════════════════════
#  Heuristique 2 : Hybride en 3 phases
# ═══════════════════════════════════════════════════════════════════

def _exhaustive_phase(instance, max_size):
    """Phase 1 : Énumère exhaustivement toutes les configurations
    élémentaires de taille <= max_size.

    Garantit de ne rater aucune configuration courte, qui sont
    souvent les plus utiles pour le programme linéaire.
    """
    configs = set()
    for size in range(1, max_size + 1):
        for combo in combinations(range(instance.num_sensors), size):
            s = frozenset(combo)
            if _is_valid_configuration(instance, s) and _is_elementary(instance, s):
                configs.add(s)
    return configs


def _zone_driven_phase(instance, configs, target_count, max_attempts_factor=50):
    """Phase 2 : Sampling structurel zone-driven.

    À chaque tentative :
    1. Interdit ~25% des capteurs (forçage de diversité)
    2. Choisit une zone non couverte ALÉATOIRE comme cible
    3. Sélectionne le capteur qui couvre cette zone ET maximise
       la couverture résiduelle
    4. Réduit vers l'élémentaire

    Brise la convergence vers les mêmes capteurs dominants,
    contrairement au greedy classique.
    """
    attempts = 0
    max_attempts = (target_count - len(configs)) * max_attempts_factor

    while len(configs) < target_count and attempts < max_attempts:
        attempts += 1
        uncovered = set(range(instance.num_zones))
        current = []

        # Interdire un sous-ensemble aléatoire de capteurs
        k_ban = max(0, instance.num_sensors // 4)
        banned = set(random.sample(range(instance.num_sensors), k=k_ban))

        while uncovered:
            # Choisir une zone cible aléatoire (pas la plus "facile")
            target_zone = random.choice(list(uncovered))

            candidates = [
                s for s in range(instance.num_sensors)
                if s not in banned
                and s not in current
                and target_zone in instance.get_zones_covered_by(s)
            ]

            if not candidates:
                break  # Ban trop agressif → abandonner cette tentative

            # Parmi les candidats, prendre le plus couvrant
            best = max(candidates,
                       key=lambda s: len(instance.get_zones_covered_by(s) & uncovered))
            uncovered -= instance.get_zones_covered_by(best)
            current.append(best)

        if uncovered:
            continue

        configs.add(_reduce_to_elementary(instance, set(current)))

    return configs


def _random_greedy_phase(instance, configs, target_count, max_attempts_factor=100):
    """Phase 3 : Random greedy classique, utilisé en filet de sécurité
    si les phases 1 et 2 n'ont pas atteint le quota demandé."""
    attempts = 0
    max_attempts = (target_count - len(configs)) * max_attempts_factor

    while len(configs) < target_count and attempts < max_attempts:
        attempts += 1
        uncovered = set(range(instance.num_zones))
        current = []
        sensors = list(range(instance.num_sensors))
        random.shuffle(sensors)

        for s in sensors:
            new = instance.get_zones_covered_by(s) & uncovered
            if new:
                uncovered -= new
                current.append(s)
            if not uncovered:
                break

        if uncovered:
            continue

        configs.add(_reduce_to_elementary(instance, set(current)))

    return configs


def generate_greedy_max_coverage(instance, num_configs):
    """Heuristique 2 : Hybride 3 phases (exhaustif + zone-driven + random greedy).

    Phase 1 — Exhaustif :
        Énumère toutes les configurations élémentaires jusqu'à une taille
        calibrée selon N. Garantit l'optimalité sur les petites instances
        et ne rate jamais les configurations courtes (souvent les meilleures).
        Seuil : taille 4 si N<=30, taille 3 si N<=60, taille 2 sinon.

    Phase 2 — Zone-driven :
        Sampling structurel avec bannissement aléatoire de capteurs et
        sélection par zone cible aléatoire. Brise la convergence vers
        les mêmes capteurs dominants, contrairement au greedy classique.

    Phase 3 — Random greedy (fallback) :
        Complète le quota si les phases précédentes sont épuisées.
    """
    n = instance.num_sensors

    # Calibrage automatique du seuil d'exhaustivité
    if n <= 30:
        exhaustive_max_size = 4
    elif n <= 60:
        exhaustive_max_size = 3
    else:
        exhaustive_max_size = 0  # Pas d'exhaustif, direct zone-driven

    # Phase 1 : Exhaustif
    configs = _exhaustive_phase(instance, exhaustive_max_size)

    if len(configs) >= num_configs:
        return [tuple(sorted(c)) for c in configs]

    # Phase 2 : Zone-driven
    configs = _zone_driven_phase(instance, configs, num_configs)

    # Phase 3 : Fallback random greedy
    if len(configs) < num_configs:
        configs = _random_greedy_phase(instance, configs, num_configs)

    return [tuple(sorted(c)) for c in configs]


# ═══════════════════════════════════════════════════════════════════
#  Wrapper public (interface inchangée pour main.py / analysis.py)
# ═══════════════════════════════════════════════════════════════════

def generate_elementary_configurations(instance, num_configs, method="random"):
    """Fonction wrapper — interface publique inchangée.

    Args:
        instance   : SensorNetworkInstance
        num_configs: nombre de configurations souhaitées
        method     : "random"  → heuristique gloutonne aléatoire (Partie 2)
                     "greedy"  → hybride 3 phases (Partie 2, méthode améliorée)

    Returns:
        Liste de tuples (indices de capteurs triés, 0-based).
    """
    if method == "greedy":
        return generate_greedy_max_coverage(instance, num_configs)
    else:
        return generate_random_greedy(instance, num_configs)


def display_configurations(instance, configurations):
    """Affiche les configurations élémentaires de manière lisible."""
    print(f"\n  {len(configurations)} configurations élémentaires générées :\n")
    for idx, config in enumerate(configurations):
        sensors_str = ", ".join(f"s{s + 1}" for s in config)
        print(f"    u{idx + 1} = ({sensors_str})")
    print()
