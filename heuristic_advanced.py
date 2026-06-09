"""
heuristic_advanced.py — Méthodes d'optimisation avancées pour la génération
de configurations élémentaires.

Trois nouvelles approches, du plus simple au plus sophistiqué :

1. RandomPruningGreedy
   Génération gloutonne avec scoring dynamique et pruning probabiliste.
   À chaque étape, le capteur suivant est tiré selon une distribution de
   probabilités basée sur le ratio (zones_nouvelles / durée_de_vie).
   Après construction, un pruning aléatoire pondéré retire les capteurs
   les moins indispensables.

2. DualGuidedGenerator (LP-guided)
   Après une première résolution LP, les prix duaux des contraintes de
   durée de vie révèlent quels capteurs sont les goulots d'étranglement.
   On biaise la génération suivante pour économiser ces capteurs : un
   capteur avec un prix dual élevé (contrainte binding) reçoit un malus
   dans le scoring greedy, ce qui force la diversification.

3. ColumnGeneration
   Implémentation complète de la génération de colonnes (méthode exacte) :
   - On résout le LP maître restreint sur un sous-ensemble de configs
   - On cherche la colonne (config) de coût réduit le plus négatif via
     un sous-problème de pricing (knapsack 0-1 approché par greedy)
   - Si une config améliorante est trouvée, on l'ajoute et on itère
   Cette méthode garantit l'optimalité du LP dans la limite du nb d'itérations.
"""

import random
import math
from itertools import combinations
from lp_solver import solve_with_scipy, solve_with_duals


# ═══════════════════════════════════════════════════════════════════
#  Utilitaires communs
# ═══════════════════════════════════════════════════════════════════

def _covers_all(instance, config):
    covered = set()
    for s in config:
        covered.update(instance.get_zones_covered_by(s))
    return len(covered) == instance.num_zones


def _is_elementary(instance, config):
    config = list(config)
    for i in range(len(config)):
        if _covers_all(instance, config[:i] + config[i+1:]):
            return False
    return True


def _reduce_to_elementary(instance, config, shuffle=True):
    """Réduit gloutonement vers une config élémentaire.
    
    shuffle=True : ordre aléatoire → diversité
    shuffle=False : retire les capteurs par ordre croissant de zones couvertes
                    → tend à garder les capteurs très couvrants
    """
    sensors = list(config)
    if shuffle:
        random.shuffle(sensors)
    else:
        # Tri croissant : on essaie d'abord de retirer les petits capteurs
        sensors.sort(key=lambda s: len(instance.get_zones_covered_by(s)))

    elementary = set(config)
    for s in sensors:
        if _covers_all(instance, elementary - {s}):
            elementary.remove(s)
    return frozenset(elementary)


# ═══════════════════════════════════════════════════════════════════
#  1. RandomPruningGreedy
# ═══════════════════════════════════════════════════════════════════

class RandomPruningGreedy:
    """Génération gloutonne avec scoring dynamique et pruning probabiliste.

    Scoring dynamique :
        score(s) = |zones_nouvelles(s)|^alpha / lifetime(s)^beta * (1 + epsilon)

        alpha > 1 : favorise les capteurs très couvrants
        beta  > 0 : pénalise les capteurs coûteux en énergie
        epsilon ~ Uniform(0, noise) : bruit pour la diversité

    Pruning probabiliste :
        Après construction d'une config valide, on tente de retirer chaque
        capteur avec une probabilité p_prune pondérée par son "score d'inutilité" :
            p_remove(s) = softmax(-score_contribution(s))
        Cela favorise le retrait des capteurs qui contribuent peu mais consomment
        beaucoup, plutôt qu'un retrait uniforme aléatoire.
    """

    def __init__(self, alpha=1.5, beta=0.5, noise=0.3, p_prune=0.4,
                 temperature=1.0):
        """
        Args:
            alpha      : exposant sur la couverture (>=1 pour favoriser la couverture)
            beta       : exposant sur la durée de vie (>0 pour économiser l'énergie)
            noise      : amplitude du bruit uniforme sur les scores
            p_prune    : probabilité de tenter le pruning sur chaque capteur
            temperature: température du softmax pour le pruning (plus haut = plus uniforme)
        """
        self.alpha = alpha
        self.beta = beta
        self.noise = noise
        self.p_prune = p_prune
        self.temperature = temperature

    def _build_one(self, instance):
        """Construit une configuration complète par scoring dynamique."""
        uncovered = set(range(instance.num_zones))
        current = []
        available = set(range(instance.num_sensors))

        while uncovered and available:
            # Calculer les scores pour tous les capteurs disponibles
            scored = []
            for s in available:
                new_zones = instance.get_zones_covered_by(s) & uncovered
                if not new_zones:
                    continue  # Ne contribue rien : skip
                coverage_gain = len(new_zones)
                lifetime = max(instance.lifetimes[s], 1e-6)
                # Score : couverture^alpha / durée^beta + bruit
                score = (coverage_gain ** self.alpha) / (lifetime ** self.beta)
                score *= (1.0 + random.uniform(0, self.noise))
                scored.append((s, score))

            if not scored:
                break

            # Tirage probabiliste : softmax des scores
            max_score = max(s for _, s in scored)
            weights = [math.exp((s - max_score) / max(self.temperature, 1e-9))
                       for _, s in scored]
            total = sum(weights)
            r = random.random() * total
            cumul = 0.0
            chosen = scored[0][0]
            for (s, _), w in zip(scored, weights):
                cumul += w
                if r <= cumul:
                    chosen = s
                    break

            new_zones = instance.get_zones_covered_by(chosen) & uncovered
            uncovered -= new_zones
            current.append(chosen)
            available.discard(chosen)

        if uncovered:
            return None  # Échec (ne devrait pas arriver si l'instance est valide)
        return current

    def _probabilistic_prune(self, instance, config):
        """Pruning probabiliste : retire les capteurs avec une proba pondérée."""
        sensors = list(config)
        random.shuffle(sensors)

        current = set(config)
        for s in sensors:
            if random.random() > self.p_prune:
                continue
            # Tenter de retirer s
            candidate = current - {s}
            if candidate and _covers_all(instance, candidate):
                current = candidate

        return current

    def generate(self, instance, num_configs, max_attempts_factor=50):
        """Génère num_configs configurations élémentaires.

        Returns:
            list of frozensets
        """
        configs = set()
        attempts = 0
        max_attempts = num_configs * max_attempts_factor

        while len(configs) < num_configs and attempts < max_attempts:
            attempts += 1

            raw = self._build_one(instance)
            if raw is None:
                continue

            # Pruning probabiliste d'abord
            pruned = self._probabilistic_prune(instance, set(raw))

            # Réduction déterministe pour garantir l'élémentarité
            elementary = _reduce_to_elementary(instance, pruned, shuffle=True)

            if _covers_all(instance, elementary):
                configs.add(elementary)

        return list(configs)


# ═══════════════════════════════════════════════════════════════════
#  2. DualGuidedGenerator
# ═══════════════════════════════════════════════════════════════════

class DualGuidedGenerator:
    """Génération guidée par les prix duaux du LP.

    Principe :
        1. On résout un LP initial sur un petit ensemble de configurations.
        2. Les prix duaux λ_i indiquent la "rareté" de chaque capteur i :
           λ_i > 0 ↔ la contrainte lifetime[i] est binding ↔ capteur saturé.
        3. On pénalise ces capteurs dans le scoring greedy :
               score(s) = couverture / (lifetime[s] * (1 + dual_weight * λ_s))
           Ce qui pousse le générateur à éviter les capteurs déjà sur-utilisés
           et à explorer des configurations plus diversifiées.
        4. On itère : chaque batch de nouvelles configs est injecté dans le LP,
           les duaux sont mis à jour, etc.

    C'est une version simplifiée de la génération de colonnes guidée par les duaux.
    """

    def __init__(self, dual_weight=2.0, noise=0.2, iterations=5):
        """
        Args:
            dual_weight : intensité de la pénalisation duale (plus grand = plus fort évitement)
            noise       : bruit aléatoire sur les scores (pour la diversité)
            iterations  : nombre de rounds LP → génération → LP
        """
        self.dual_weight = dual_weight
        self.noise = noise
        self.iterations = iterations

    def _score_sensor(self, instance, sensor, uncovered, dual_prices):
        """Score d'un capteur dans le contexte courant."""
        new_zones = instance.get_zones_covered_by(sensor) & uncovered
        if not new_zones:
            return -1.0

        coverage = len(new_zones)
        lifetime = max(instance.lifetimes[sensor], 1e-6)
        dual = dual_prices.get(sensor, 0.0)

        # Pénalité duale : on coûte plus cher si on est un goulot
        effective_cost = lifetime * (1.0 + self.dual_weight * max(dual, 0.0))
        score = coverage / effective_cost
        score *= (1.0 + random.uniform(0, self.noise))
        return score

    def _build_one(self, instance, dual_prices):
        """Construit une configuration en évitant les capteurs à fort prix dual."""
        uncovered = set(range(instance.num_zones))
        current = []
        available = set(range(instance.num_sensors))

        while uncovered and available:
            best_s, best_score = None, -1.0
            for s in available:
                sc = self._score_sensor(instance, s, uncovered, dual_prices)
                if sc > best_score:
                    best_score, best_s = sc, s

            if best_s is None:
                break

            current.append(best_s)
            uncovered -= instance.get_zones_covered_by(best_s)
            available.discard(best_s)

        if uncovered:
            return None
        return set(current)

    def generate(self, instance, num_configs, seed_configs=None):
        """Génère num_configs configurations par itérations LP+dual.

        Args:
            instance    : SensorNetworkInstance
            num_configs : nombre de configurations cibles
            seed_configs: configurations initiales (optionnel)
        """
        # Initialisation : configs seeds ou random greedy minimal
        configs = set()
        if seed_configs:
            configs.update(frozenset(c) for c in seed_configs)

        # Générer quelques configs initiales si nécessaire
        if len(configs) < 5:
            _bootstrap(instance, configs, target=max(5, num_configs // 10))

        configs_list = [tuple(sorted(c)) for c in configs]
        dual_prices = {}

        per_iter = max(1, (num_configs - len(configs)) // self.iterations)

        for iteration in range(self.iterations):
            if len(configs) >= num_configs:
                break

            # Résoudre le LP et récupérer les prix duaux
            if configs_list:
                obj, variables, dual_prices = solve_with_duals(instance, configs_list)

            # Générer de nouvelles configs guidées par les duaux
            new_in_iter = 0
            attempts = 0
            max_attempts = per_iter * 100

            while new_in_iter < per_iter and attempts < max_attempts:
                attempts += 1
                raw = self._build_one(instance, dual_prices)
                if raw is None:
                    continue
                elementary = _reduce_to_elementary(instance, raw, shuffle=True)
                if elementary not in configs:
                    configs.add(elementary)
                    configs_list.append(tuple(sorted(elementary)))
                    new_in_iter += 1

        # Compléter si besoin avec du random pur
        _bootstrap(instance, configs, target=num_configs)

        return [tuple(sorted(c)) for c in configs]


# ═══════════════════════════════════════════════════════════════════
#  3. ColumnGeneration
# ═══════════════════════════════════════════════════════════════════

class ColumnGeneration:
    """Génération de colonnes (Branch & Price light).

    Algorithme :
        1. Initialiser avec un ensemble restreint de configurations
        2. Résoudre le LP maître restreint → obtenir les prix duaux λ_i
        3. Résoudre le sous-problème de pricing :
               Trouver une configuration C qui maximise :
                   reduced_cost(C) = 1 - sum(λ_i for i in C)
           Si reduced_cost > ε, la config C améliore le LP → l'ajouter
        4. Itérer jusqu'à convergence ou max_iter

    Le sous-problème de pricing est un problème de couverture pondérée.
    On le résout de deux façons :
        - Exact (petites instances) : énumération par taille croissante
        - Approché (grandes instances) : greedy sur le coût réduit marginal

    Note : Cette méthode converge vers la solution LP exacte (borne supérieure
    du problème entier). Elle est particulièrement utile quand le nombre de
    configurations nécessaires est petit.
    """

    def __init__(self, max_iter=200, pricing_mode="auto", epsilon=1e-6,
                 fallback_random=True):
        """
        Args:
            max_iter       : nombre max d'itérations de génération de colonnes
            pricing_mode   : "exact" / "greedy" / "auto" (exact si N<=50, sinon greedy)
            epsilon        : seuil de coût réduit pour stopper (convergence)
            fallback_random: si True, complète avec du random si convergence prématurée
        """
        self.max_iter = max_iter
        self.pricing_mode = pricing_mode
        self.epsilon = epsilon
        self.fallback_random = fallback_random

    def _solve_pricing_greedy(self, instance, dual_prices):
        """Sous-problème de pricing : greedy.

        Cherche une configuration de coût réduit maximal :
            reduced_cost = 1 - sum(λ_i for i in config)

        Un capteur avec λ_i élevé est "cher" dans le sous-problème.
        On veut couvrir toutes les zones en minimisant sum(λ_i).

        Stratégie : à chaque étape, on choisit le capteur qui maximise le ratio
            |nouvelles zones| / (λ_i + epsilon)
        ce qui correspond à un greedy sur le knapsack fractionnel.
        """
        uncovered = set(range(instance.num_zones))
        config = set()
        available = set(range(instance.num_sensors))
        total_dual_cost = 0.0

        while uncovered and available:
            best_s, best_ratio = None, -1.0
            for s in available:
                new = instance.get_zones_covered_by(s) & uncovered
                if not new:
                    continue
                dual_cost = max(dual_prices.get(s, 0.0), 0.0)
                # Ratio : zones nouvelles / coût dual marginal
                ratio = len(new) / (dual_cost + 1e-9)
                # Bruit pour diversité
                ratio *= (1.0 + random.uniform(0, 0.1))
                if ratio > best_ratio:
                    best_ratio, best_s = ratio, s

            if best_s is None:
                break
            uncovered -= instance.get_zones_covered_by(best_s)
            total_dual_cost += max(dual_prices.get(best_s, 0.0), 0.0)
            config.add(best_s)
            available.discard(best_s)

        if uncovered:
            return None, None

        reduced_cost = 1.0 - total_dual_cost
        return config, reduced_cost

    def _solve_pricing_exact(self, instance, dual_prices, max_size=None):
        """Sous-problème de pricing : exact par énumération.

        Pour chaque combinaison de taille croissante, calcule le coût réduit.
        Retourne la combinaison avec le coût réduit maximal (le plus positif).
        """
        if max_size is None:
            max_size = min(instance.num_sensors, 8)  # Limite combinatoire

        best_config = None
        best_rc = -float('inf')

        for size in range(1, max_size + 1):
            for combo in combinations(range(instance.num_sensors), size):
                s = frozenset(combo)
                if not _covers_all(instance, s):
                    continue
                if not _is_elementary(instance, s):
                    continue
                dual_cost = sum(max(dual_prices.get(i, 0.0), 0.0) for i in s)
                rc = 1.0 - dual_cost
                if rc > best_rc:
                    best_rc, best_config = rc, s

            # Élagage : si on a déjà trouvé une bonne config et qu'on grossit,
            # le coût réduit ne peut qu'empirer (plus de capteurs = plus de coût dual)
            if best_rc > self.epsilon and size >= 2:
                min_possible_cost = min(
                    max(dual_prices.get(s, 0.0), 0.0)
                    for s in range(instance.num_sensors)
                ) * (size + 1)
                if 1.0 - min_possible_cost <= self.epsilon:
                    break

        return best_config, best_rc

    def _choose_pricing_mode(self, instance):
        if self.pricing_mode == "exact":
            return "exact"
        if self.pricing_mode == "greedy":
            return "greedy"
        # Auto : exact pour petites instances, greedy pour grandes
        return "exact" if instance.num_sensors <= 40 else "greedy"

    def generate(self, instance, num_configs, seed_configs=None):
        """Lance la génération de colonnes.

        Args:
            instance    : SensorNetworkInstance
            num_configs : nombre maximum de configurations à générer
            seed_configs: configurations de démarrage (optionnel)

        Returns:
            list of tuples (configurations élémentaires)
        """
        configs = set()

        # Initialisation
        if seed_configs:
            configs.update(frozenset(c) for c in seed_configs)
        _bootstrap(instance, configs, target=max(3, min(10, num_configs // 5)))

        configs_list = [tuple(sorted(c)) for c in configs]
        mode = self._choose_pricing_mode(instance)

        history = []  # (iteration, obj, nb_configs)
        obj = 0.0

        for iteration in range(self.max_iter):
            if len(configs) >= num_configs:
                break

            # Résoudre le LP maître restreint
            obj, variables, dual_prices = solve_with_duals(instance, configs_list)

            if not dual_prices:
                break

            # Résoudre le sous-problème de pricing
            if mode == "exact":
                new_config, rc = self._solve_pricing_exact(instance, dual_prices)
            else:
                # Plusieurs tentatives greedy pour diversité
                best_config_found, best_rc = None, -float('inf')
                for _ in range(5):
                    c, r = self._solve_pricing_greedy(instance, dual_prices)
                    if c is not None and r is not None and r > best_rc:
                        best_rc, best_config_found = r, c
                new_config, rc = best_config_found, best_rc

            history.append((iteration, obj, len(configs)))

            # Condition d'arrêt : plus de colonne améliorante
            if new_config is None or rc is None or rc <= self.epsilon:
                break

            # Réduire à élémentaire et ajouter
            elementary = _reduce_to_elementary(instance, new_config, shuffle=False)
            if elementary not in configs:
                configs.add(elementary)
                configs_list.append(tuple(sorted(elementary)))

        # Compléter avec du random si demandé et quota non atteint
        if self.fallback_random and len(configs) < num_configs:
            _bootstrap(instance, configs, target=num_configs)

        return [tuple(sorted(c)) for c in configs], history, obj


# ═══════════════════════════════════════════════════════════════════
#  Utilitaire interne : bootstrap random greedy
# ═══════════════════════════════════════════════════════════════════

def _bootstrap(instance, configs_set, target, max_attempts_factor=100):
    """Remplit configs_set jusqu'à target avec du random greedy simple."""
    needed = target - len(configs_set)
    if needed <= 0:
        return

    attempts = 0
    max_attempts = needed * max_attempts_factor

    while len(configs_set) < target and attempts < max_attempts:
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

        configs_set.add(_reduce_to_elementary(instance, set(current), shuffle=True))


# ═══════════════════════════════════════════════════════════════════
#  4. SimulatedAnnealingGenerator
# ═══════════════════════════════════════════════════════════════════

class SimulatedAnnealingGenerator:
    """Génération par Recuit Simulé.
    
    Améliore itérativement des configurations élémentaires en tentant
    des échanges de capteurs. La fonction de coût favorise les capteurs
    avec une grande durée de vie et pénalise les capteurs déjà beaucoup
    utilisés dans les configurations précédentes pour forcer la diversité.
    """

    def __init__(self, initial_temp=10.0, cooling_rate=0.95, min_temp=0.1, 
                 steps_per_temp=10, penalty_weight=1.0):
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.min_temp = min_temp
        self.steps_per_temp = steps_per_temp
        self.penalty_weight = penalty_weight

    def _cost(self, instance, config, usage_counts):
        cost = 0.0
        for s in config:
            lifetime = max(instance.lifetimes[s], 1e-6)
            penalty = usage_counts.get(s, 0) * self.penalty_weight
            cost += (1.0 / lifetime) + penalty
        return cost

    def _get_neighbor(self, instance, config):
        """Génère un voisin en retirant un capteur et en réparant."""
        if len(config) <= 1:
            return config
            
        sensors = list(config)
        # Retirer un capteur aléatoire
        dropped = random.choice(sensors)
        sensors.remove(dropped)
        
        # Identifier les zones orphelines
        covered = set()
        for s in sensors:
            covered.update(instance.get_zones_covered_by(s))
            
        uncovered = set(range(instance.num_zones)) - covered
        
        # Réparer gloutonnement
        available = list(set(range(instance.num_sensors)) - set(sensors) - {dropped})
        random.shuffle(available)
        
        current = list(sensors)
        for s in available:
            if not uncovered:
                break
            new_zones = instance.get_zones_covered_by(s) & uncovered
            if new_zones:
                uncovered -= new_zones
                current.append(s)
                
        if uncovered:
            # Si on n'arrive pas à réparer (très rare), on retourne l'original
            return config
            
        return _reduce_to_elementary(instance, set(current), shuffle=True)

    def generate(self, instance, num_configs):
        configs = set()
        usage_counts = {}
        
        # Initialisation rapide avec quelques configs aléatoires
        _bootstrap(instance, configs, target=min(5, num_configs))
        for c in configs:
            for s in c:
                usage_counts[s] = usage_counts.get(s, 0) + 1
                
        attempts = 0
        max_attempts = num_configs * 50
        
        while len(configs) < num_configs and attempts < max_attempts:
            attempts += 1
            
            # Prendre une configuration de départ (aléatoire parmi celles existantes ou nouvelle)
            if configs and random.random() < 0.5:
                current_config = random.choice(list(configs))
            else:
                tmp_set = set()
                _bootstrap(instance, tmp_set, target=1)
                if not tmp_set:
                    continue
                current_config = list(tmp_set)[0]
                
            current_cost = self._cost(instance, current_config, usage_counts)
            best_config = current_config
            best_cost = current_cost
            
            temp = self.initial_temp
            
            while temp > self.min_temp:
                for _ in range(self.steps_per_temp):
                    neighbor = self._get_neighbor(instance, best_config)
                    neighbor_cost = self._cost(instance, neighbor, usage_counts)
                    
                    delta = neighbor_cost - current_cost
                    
                    if delta < 0 or random.random() < math.exp(-delta / temp):
                        current_config = neighbor
                        current_cost = neighbor_cost
                        
                        if current_cost < best_cost:
                            best_config = current_config
                            best_cost = current_cost
                            
                temp *= self.cooling_rate
                
            if best_config not in configs:
                configs.add(best_config)
                for s in best_config:
                    usage_counts[s] = usage_counts.get(s, 0) + 1

        # Fallback si on manque de configs
        _bootstrap(instance, configs, target=num_configs)
        
        return [tuple(sorted(c)) for c in configs]

# ═══════════════════════════════════════════════════════════════════
#  5. RealTimeGreedyGenerator
# ═══════════════════════════════════════════════════════════════════

class RealTimeGreedyGenerator:
    """Glouton Temps-Réel (Énergie Résiduelle).
    
    Génère des configurations en simulant l'usure des batteries au fur
    et à mesure. Les capteurs dont l'énergie tombe à zéro ne peuvent plus
    être utilisés, ce qui force naturellement la diversité.
    """

    def __init__(self, step_time=1.0):
        self.step_time = step_time

    def generate(self, instance, num_configs):
        configs = set()
        
        # Copie locale de l'énergie de chaque capteur
        residual_energy = list(instance.lifetimes)
        
        attempts = 0
        max_attempts = num_configs * 50
        
        while len(configs) < num_configs and attempts < max_attempts:
            attempts += 1
            
            # Filtre des capteurs encore vivants
            available_sensors = [s for s in range(instance.num_sensors) if residual_energy[s] > 0]
            
            # Vérifier si les capteurs vivants couvrent encore toutes les zones
            covered_zones = set()
            for s in available_sensors:
                covered_zones.update(instance.get_zones_covered_by(s))
                
            if len(covered_zones) < instance.num_zones:
                # Reset des batteries pour pouvoir générer d'autres configurations
                residual_energy = list(instance.lifetimes)
                available_sensors = list(range(instance.num_sensors))
                
            # Random Greedy sur les capteurs vivants
            random.shuffle(available_sensors)
            
            uncovered = set(range(instance.num_zones))
            current = []
            
            for s in available_sensors:
                new = instance.get_zones_covered_by(s) & uncovered
                if new:
                    uncovered -= new
                    current.append(s)
                if not uncovered:
                    break
                    
            if uncovered:
                continue
                
            # Réduire vers l'élémentaire
            elementary = _reduce_to_elementary(instance, set(current), shuffle=True)
            
            # Déduire l'énergie
            for s in elementary:
                residual_energy[s] -= self.step_time
                
            configs.add(elementary)

        # Fallback si on n'a pas atteint le quota
        if len(configs) < num_configs:
            _bootstrap(instance, configs, target=num_configs)
            
        return [tuple(sorted(c)) for c in configs]

# ═══════════════════════════════════════════════════════════════════
#  Interface publique
# ═══════════════════════════════════════════════════════════════════

def generate_with_random_pruning(instance, num_configs,
                                  alpha=1.5, beta=0.5, noise=0.3,
                                  p_prune=0.4, temperature=1.0):
    """Interface pour RandomPruningGreedy."""
    gen = RandomPruningGreedy(alpha=alpha, beta=beta, noise=noise,
                               p_prune=p_prune, temperature=temperature)
    return [tuple(sorted(c)) for c in gen.generate(instance, num_configs)]


def generate_with_dual_guidance(instance, num_configs,
                                 dual_weight=2.0, noise=0.2, iterations=5,
                                 seed_configs=None):
    """Interface pour DualGuidedGenerator."""
    gen = DualGuidedGenerator(dual_weight=dual_weight, noise=noise, iterations=iterations)
    return gen.generate(instance, num_configs, seed_configs=seed_configs)


def generate_with_column_generation(instance, num_configs,
                                      max_iter=200, pricing_mode="auto",
                                      seed_configs=None):
    """Interface pour ColumnGeneration.

    Returns:
        (list of config tuples, history list, final_obj)
    """
    gen = ColumnGeneration(max_iter=max_iter, pricing_mode=pricing_mode)
    return gen.generate(instance, num_configs, seed_configs=seed_configs)


def generate_with_simulated_annealing(instance, num_configs,
                                      initial_temp=10.0, cooling_rate=0.95,
                                      min_temp=0.1, steps_per_temp=10, penalty_weight=1.0):
    """Interface pour SimulatedAnnealingGenerator."""
    gen = SimulatedAnnealingGenerator(initial_temp=initial_temp, cooling_rate=cooling_rate,
                                      min_temp=min_temp, steps_per_temp=steps_per_temp,
                                      penalty_weight=penalty_weight)
    return gen.generate(instance, num_configs)

def generate_with_real_time_greedy(instance, num_configs, step_time=1.0):
    """Interface pour RealTimeGreedyGenerator."""
    gen = RealTimeGreedyGenerator(step_time=step_time)
    return gen.generate(instance, num_configs)
