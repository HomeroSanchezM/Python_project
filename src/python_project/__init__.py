import random

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class AminoAcid:
    """Représente un acide aminé"""
    name: str
    code_3: str
    code_1: str


# Liste des acides aminés communs
AMINO_ACIDS = [
    AminoAcid("Alanine", "Ala", "A" ),
    AminoAcid("Arginine", "Arg", "R" ),
    AminoAcid("Asparagine", "Asn", "N" ),
    AminoAcid("Acide aspartique", "Asp", "D" ),
    AminoAcid("Cystéine", "Cys", "C"),
    AminoAcid("Acide glutamique", "Glu", "E" ),
    AminoAcid("Glutamine", "Gln", "Q"),
    AminoAcid("Glycine", "Gly", "G" ),
    AminoAcid("Histidine", "His", "H"),
    AminoAcid("Isoleucine", "Ile", "I" ),
    AminoAcid("Leucine", "Leu", "L" ),
    AminoAcid("Lysine", "Lys", "K" ),
    AminoAcid("Méthionine", "Met", "M"),
    AminoAcid("Phénylalanine", "Phe", "F"),
    AminoAcid("Proline", "Pro", "P"),
    AminoAcid("Sérine", "Ser", "S"),
    AminoAcid("Thréonine", "Thr", "T"),
    AminoAcid("Tryptophane", "Trp", "W"),
    AminoAcid("Tyrosine", "Tyr", "Y"),
    AminoAcid("Valine", "Val", "V")
]

#Liste pour déterminer si un AA est deutérable (True) ou non (False)
restrictions = [
        True,   # Ala
        False,  # Arg
        True,   # Asn
        True,   # Asp
        False,  # Cys
        False,  # Glu
        True,   # Gln
        True,   # Gly
        False,  # His
        True,   # Ile
        True,   # Leu
        True,   # Lys
        True,   # Met
        True,   # Phe
        True,   # Pro
        True,   # Ser
        True,   # Thr
        True,   # Trp
        True,   # Typ
        True,   # Val
    ]


class Chromosome:
    """Représente un chromosome dans l'algorithme génétique"""

    def __init__(self, aa_list: List[AminoAcid], modifiable: List[bool] = None):
        """
        Initialise un chromosome

        Args:
            aa_list: Liste des acides aminés
            modifiable: Liste indiquant quels AA peuvent être modifiés
        """
        self.aa_list = aa_list
        self.n_aa = len(aa_list)

        # Si pas de restrictions spécifiées, tous sont modifiables
        if modifiable is None:
            self.modifiable = [True] * self.n_aa
        else:
            assert len(modifiable) == self.n_aa, "modifiable doit avoir la même taille que aa_list"
            self.modifiable = modifiable

        # Vecteur de deutération (True = deutéré, False = non deutéré)
        self.deuteration = [False] * self.n_aa

        # Fitness du chromosome
        self.fitness = 0.0

        self.d2o = int(50)

    def randomize(self):
        """
        Initialise aléatoirement le vecteur de deutération (respectant les restrictions)
        Initialise le pourcentage de d2o avec des variation entre -5 et +5
        """
        for i in range(self.n_aa):
            if self.modifiable[i]:
                self.deuteration[i] = random.choice([True, False])
            else:
                self.deuteration[i] = False
        if self.d2o > 5 and self.d2o < 95 :
            self.d2o += random.choice([-5, -4, -3 ,-2 , -1 , 0, 1, 2 , 3 , 4 , 5])


    def get_deuteration_count(self) -> int:
        """Compte le nombre d'AA deutérés"""
        return sum(self.deuteration)

    def copy(self):
        """Crée une copie du chromosome"""
        new_chrom = Chromosome(self.aa_list, self.modifiable)
        new_chrom.deuteration = self.deuteration.copy()
        new_chrom.fitness = self.fitness
        return new_chrom

    def __str__(self):
        """Représentation textuelle du chromosome"""
        result = []
        for i, aa in enumerate(self.aa_list):
            if self.deuteration[i]:
                result.append(f"{aa.code_3}(D)")
            else:
                result.append(f"{aa.code_3}(H)")
        return " | ".join(result) + " " + str(self.d2o)




class GeneticAlgorithm:
    """Algorithme génétique pour l'optimisation de la deutération"""

    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elitism: int = 2):
        """
        Initialise l'algorithme génétique

        Args:
            aa_list: Liste des acides aminés
            modifiable: Liste indiquant quels AA peuvent être modifiés
            population_size: Taille de la population
            mutation_rate: Probabilité de mutation
            crossover_rate: Probabilité de croisement
            elitism: Nombre d'individus élites préservés
        """
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.population = []
        self.best_solution = None
        self.generation = 0

    def initialize_population(self):
        """Crée la population initiale"""
        self.population = []
        for _ in range(self.population_size):
            chrom = Chromosome(self.aa_list, self.modifiable)
            chrom.randomize()
            self.population.append(chrom)

    def fitness_function(self, chromosome: Chromosome) -> float:
        """
        Calcule le fitness d'un chromosome
        Se fait avec analyses des résultats SANS
        """

    def evaluate_population(self):
        """Évalue le fitness de toute la population"""
        for chrom in self.population:
            chrom.fitness = self.fitness_function(chrom)

        # Tri par fitness décroissant
        self.population.sort(key=lambda x: x.fitness, reverse=True)

        # Mise à jour de la meilleure solution
        if self.best_solution is None or self.population[0].fitness > self.best_solution.fitness:
            self.best_solution = self.population[0].copy()

    def selection(self) -> Chromosome:
        """Sélection des chromosomes pour etre parent"""


    def crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Croisement en un point"""
        child1 = parent1.copy()
        child2 = parent2.copy()

        if random.random() < self.crossover_rate:
            # Point de croisement aléatoire
            point = random.randint(1, len(self.aa_list) - 1)

            # Échange des gènes après le point de croisement
            for i in range(point, len(self.aa_list)):
                if self.modifiable[i]:  # Respect des restrictions
                    child1.deuteration[i] = parent2.deuteration[i]
                    child2.deuteration[i] = parent1.deuteration[i]

        return child1, child2

    def mutate(self, chromosome: Chromosome):
        """
        Mutation par inversion de bit
        Il y a entre 1 et 3 mutation par vecteur
        """
        lim = random.choice([1, 2,3])
        count = 0
        for i in range(len(self.aa_list)):
            if self.modifiable[i] and random.random() < self.mutation_rate and count < lim:
                chromosome.deuteration[i] = not chromosome.deuteration[i]
                count +=1

    def evolve(self):
        """Effectue une génération d'évolution"""
        new_population = []

        # Élitisme : conservation des meilleurs individus
        for i in range(self.elitism):
            new_population.append(self.population[i].copy())

        # Création de nouveaux individus
        while len(new_population) < self.population_size:
            # Sélection des parents
            parent1 = self.selection()
            parent2 = self.selection()

            # Croisement
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation
            self.mutate(child1)
            self.mutate(child2)

            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.generation += 1

    def run(self, max_generations: int = 100):
        """
        Exécute l'algorithme génétique

        Args:
            max_generations: Nombre maximum de générations
        """
        self.initialize_population()
        self.evaluate_population()

        print(f"Génération 0: Meilleur fitness = {self.best_solution.fitness:.6f}")
        print(f"Masse = {self.best_solution.get_mass():.2f}, Cible = {self.target_mass:.2f}")

        for gen in range(max_generations):
            self.evolve()
            self.evaluate_population()

            if (gen + 1) % 10 == 0:
                print(f"Génération {gen + 1}: Meilleur fitness = {self.best_solution.fitness:.6f}")
                print(
                    f"Masse = {self.best_solution.get_mass():.2f}, Deutérés = {self.best_solution.get_deuteration_count()}")


        return self.best_solution


# Exemple d'utilisation
if __name__ == "__main__":

    # Créer et exécuter l'algorithme génétique
    ga = GeneticAlgorithm(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=10,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elitism=3
    )

    ga.initialize_population()
    for chromosome in ga.population:
        print(chromosome)

    #best = ga.run(max_generations=200)
