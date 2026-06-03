import random

class SensorNetworkInstance:
    def __init__(self, num_zones, num_sensors, lifetimes, coverage_matrix):
        self.num_zones = num_zones
        self.num_sensors = num_sensors
        self.lifetimes = lifetimes
        self.coverage_matrix = coverage_matrix

    @classmethod
    def from_file(cls, filepath):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        num_sensors = int(lines[0])
        num_zones = int(lines[1])
        lifetimes = list(map(float, lines[2].split()))
        
        coverage_matrix = [[0 for _ in range(num_zones)] for _ in range(num_sensors)]
        for i in range(num_sensors):
            if 3 + i < len(lines):
                covered_zones = list(map(int, lines[3+i].split()))
                for zone in covered_zones:
                    # Les zones sont indexées à partir de 1 dans le fichier
                    coverage_matrix[i][zone - 1] = 1
            
        return cls(num_zones, num_sensors, lifetimes, coverage_matrix)

    @classmethod
    def generate_random(cls, num_zones, num_sensors, min_lifetime=10, max_lifetime=100, coverage_prob=0.3):
        lifetimes = [random.uniform(min_lifetime, max_lifetime) for _ in range(num_sensors)]
        coverage_matrix = []
        
        for _ in range(num_sensors):
            row = [1 if random.random() < coverage_prob else 0 for _ in range(num_zones)]
            if sum(row) == 0:
                row[random.randint(0, num_zones-1)] = 1
            coverage_matrix.append(row)
        
        # S'assurer que toutes les zones sont couvertes au moins une fois
        for j in range(num_zones):
            if sum(coverage_matrix[i][j] for i in range(num_sensors)) == 0:
                coverage_matrix[random.randint(0, num_sensors-1)][j] = 1

        return cls(num_zones, num_sensors, lifetimes, coverage_matrix)
