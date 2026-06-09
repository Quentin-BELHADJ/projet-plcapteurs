import random


class SensorNetworkInstance:
    """Représente une instance du problème d'activation de capteurs."""

    def __init__(self, num_zones, num_sensors, lifetimes, coverage_matrix):
        """
        Args:
            num_zones: Nombre de zones à surveiller (M).
            num_sensors: Nombre de capteurs disponibles (N).
            lifetimes: Liste de durées de vie pour chaque capteur.
            coverage_matrix: Matrice binaire N x M. coverage_matrix[i][j] = 1
                             si le capteur i couvre la zone j.
        """
        self.num_zones = num_zones
        self.num_sensors = num_sensors
        self.lifetimes = lifetimes
        self.coverage_matrix = coverage_matrix

    @classmethod
    def from_file(cls, filepath):
        """Lit une instance depuis un fichier au format :
            Ligne 1 : nombre de capteurs (N)
            Ligne 2 : nombre de zones (M)
            Ligne 3 : N durées de vie séparées par des espaces
            Lignes 4 à N+3 : zones couvertes par chaque capteur (indices 1-based)
        """
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        num_sensors = int(lines[0])
        num_zones = int(lines[1])
        lifetimes = list(map(float, lines[2].split()))

        if len(lifetimes) != num_sensors:
            raise ValueError(
                f"Le nombre de durées de vie ({len(lifetimes)}) ne correspond pas "
                f"au nombre de capteurs ({num_sensors})."
            )

        # Construire la matrice de couverture binaire à partir des listes de zones
        coverage_matrix = []
        for i in range(num_sensors):
            row = [0] * num_zones
            if 3 + i < len(lines):
                zones_covered = list(map(int, lines[3 + i].split()))
                for z in zones_covered:
                    if 1 <= z <= num_zones:
                        row[z - 1] = 1  # Conversion 1-based → 0-based
            coverage_matrix.append(row)

        return cls(num_zones, num_sensors, lifetimes, coverage_matrix)

    @classmethod
    def from_keyboard(cls):
        """Saisie interactive d'une instance par l'utilisateur."""
        print("=== Saisie d'une instance ===")
        num_sensors = int(input("Nombre de capteurs (N) : "))
        num_zones = int(input("Nombre de zones (M) : "))

        print(f"Entrez les {num_sensors} durées de vie (séparées par des espaces) :")
        lifetimes = list(map(float, input().split()))
        while len(lifetimes) != num_sensors:
            print(f"Erreur : {len(lifetimes)} valeurs au lieu de {num_sensors}. Réessayez :")
            lifetimes = list(map(float, input().split()))

        coverage_matrix = []
        for i in range(num_sensors):
            print(f"Zones couvertes par le capteur {i + 1} (indices 1 à {num_zones}, séparés par des espaces) :")
            zones = list(map(int, input().split()))
            row = [0] * num_zones
            for z in zones:
                if 1 <= z <= num_zones:
                    row[z - 1] = 1
            coverage_matrix.append(row)

        return cls(num_zones, num_sensors, lifetimes, coverage_matrix)

    @classmethod
    def generate_random(cls, num_zones, num_sensors, min_lifetime=10, max_lifetime=100, coverage_prob=0.3):
        """Génère une instance aléatoire en garantissant la couverture de toutes les zones."""
        lifetimes = [round(random.uniform(min_lifetime, max_lifetime), 1) for _ in range(num_sensors)]
        coverage_matrix = []

        for _ in range(num_sensors):
            row = [1 if random.random() < coverage_prob else 0 for _ in range(num_zones)]
            # Chaque capteur couvre au moins une zone
            if sum(row) == 0:
                row[random.randint(0, num_zones - 1)] = 1
            coverage_matrix.append(row)

        # S'assurer que toutes les zones sont couvertes par au moins un capteur
        for j in range(num_zones):
            if sum(coverage_matrix[i][j] for i in range(num_sensors)) == 0:
                coverage_matrix[random.randint(0, num_sensors - 1)][j] = 1

        return cls(num_zones, num_sensors, lifetimes, coverage_matrix)

    def get_zones_covered_by(self, sensor_idx):
        """Retourne l'ensemble des zones couvertes par un capteur donné (indices 0-based)."""
        return {j for j in range(self.num_zones) if self.coverage_matrix[sensor_idx][j] == 1}

    def display(self):
        """Affiche un résumé lisible de l'instance."""
        print(f"\n{'='*60}")
        print(f"  Instance : {self.num_sensors} capteurs, {self.num_zones} zones")
        print(f"{'='*60}")
        print(f"\n{'Capteur':<10} {'Zones couvertes':<35} {'Durée de vie':>12}")
        print(f"{'-'*10} {'-'*35} {'-'*12}")
        for i in range(self.num_sensors):
            zones = [str(j + 1) for j in range(self.num_zones) if self.coverage_matrix[i][j] == 1]
            zones_str = ", ".join(zones) if zones else "(aucune)"
            # Tronquer si trop long
            if len(zones_str) > 33:
                zones_str = zones_str[:30] + "..."
            print(f"  s{i+1:<7} {zones_str:<35} {self.lifetimes[i]:>10.1f}")
        print()
