"""
Module de génération de population pour l'optimisation de deutération
=====================================================================

Ce module gère la création et l'évolution de populations de chromosomes
représentant des configurations de deutération d'acides aminés.

Architecture:
    - AminoAcid: Dataclass représentant un acide aminé
    - Chromosome: Classe représentant une solution (configuration de deutération)
    - PopulationGenerator: Classe principale gérant la génération/évolution

Usage:
    # Première génération
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=100,
        d2o_initial=50,
        elitism=3
    )
    population = generator.generate_initial_population()

    # Générations suivantes
    new_population = generator.generate_next_generation(
        previous_population=population,
        fitness_scores=[0.8, 0.7, ...],
        mutation_rate=0.15,
        crossover_rate=0.8,
        d2o_variation_rate=5
    )
"""

import random
import argparse
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

# ============================================================================
#                           PARSING DES ARGUMENTS
# ============================================================================

def parse_arguments():
    """
    Parse les arguments de ligne de commande pour configurer l'algorithme génétique.

    Returns:
        argparse.Namespace: Arguments parsés
    """
    parser = argparse.ArgumentParser(
        description="Algorithme génétique pour l'optimisation de la deutération d'acides aminés",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python __init__.py --population_size 50 --d2o_initial 60
  python __init__.py --mutation_rate 0.2 --crossover_rate 0.9 --elitism 5
  python __init__.py -p 100 -d 50 -e 3 -m 0.15 -c 0.8 -v 10
        """
    )

    # Paramètres de population
    parser.add_argument(
        '-p', '--population_size',
        type=int,
        default=10,
        help='Taille de la population (nombre de chromosomes). Défaut: 10'
    )

    parser.add_argument(
        '-e', '--elitism',
        type=int,
        default=2,
        help='Nombre d\'individus élites préservés à chaque génération. Défaut: 2'
    )

    # Paramètres D2O
    parser.add_argument(
        '-d', '--d2o_initial',
        type=int,
        default=50,
        help='Pourcentage initial de D2O (0-100). Utilisé pour la génération 0. Défaut: 50'
    )

    parser.add_argument(
        '-v', '--d2o_variation_rate',
        type=int,
        default=5,
        help='Amplitude maximale de variation du D2O (ex: 5 pour ±5%%). Défaut: 5'
    )

    # Paramètres génétiques
    parser.add_argument(
        '-m', '--mutation_rate',
        type=float,
        default=0.15,
        help='Taux de mutation (0.0-1.0). Probabilité qu\'un gène soit muté. Défaut: 0.15'
    )

    parser.add_argument(
        '-c', '--crossover_rate',
        type=float,
        default=0.8,
        help='Taux de croisement (0.0-1.0). Probabilité de croisement entre deux parents. Défaut: 0.8'
    )

    # Paramètres d'exécution
    parser.add_argument(
        '-g', '--generations',
        type=int,
        default=1,
        help='Nombre de générations à exécuter. Défaut: 1'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Graine aléatoire pour la reproductibilité. Défaut: None (aléatoire)'
    )

    args = parser.parse_args()

    # Validation des arguments
    if args.population_size <= 0:
        parser.error("population_size doit être > 0")

    if not (0 <= args.d2o_initial <= 100):
        parser.error("d2o_initial doit être entre 0 et 100")

    if args.elitism < 0:
        parser.error("elitism doit être >= 0")

    if args.elitism >= args.population_size:
        parser.error("elitism doit être < population_size")

    if not (0 <= args.d2o_variation_rate <= 100):
        parser.error("d2o_variation_rate doit être entre 0 et 100")

    if not (0 <= args.mutation_rate <= 1):
        parser.error("mutation_rate doit être entre 0.0 et 1.0")

    if not (0 <= args.crossover_rate <= 1):
        parser.error("crossover_rate doit être entre 0.0 et 1.0")

    if args.generations < 1:
        parser.error("generations doit être >= 1")

    return args

# ============================================================================
#                   DÉFINITION DES STRUCTURES DE DONNÉES
# ============================================================================

@dataclass
class AminoAcid:
    """
    Représente un acide aminé avec ses codes nomenclature.

    Attributes:
        name (str): Nom complet de l'acide aminé (ex: "Alanine")
        code_3 (str): Code à 3 lettres (ex: "Ala")
        code_1 (str): Code à 1 lettre (ex: "A")
    """
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

# ============================================================================
#                           CLASSE CHROMOSOME
# ============================================================================

class Chromosome:
    """
    Représente un chromosome dans l'algorithme génétique

    Un chromosome encode:
        - Quels acides aminés sont deutérés (vecteur booléen)
        - Le pourcentage de D2O utilisé (int entre 0 et 100)
        - Le score de fitness (calculé à partir des données SANS)

    Attributes:
        aa_list (List[AminoAcid]): Liste des acides aminés
        modifiable (List[bool]): Restrictions de deutération par AA
        deuteration (List[bool]): Vecteur de deutération (True = deutéré)
        d2o (int): Pourcentage de D2O (0-100)
        fitness (float): Score de fitness du chromosome

    """

    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 d2o_initial: int = 50):
        """
        Initialise un chromosome

        Args:
            aa_list: Liste des acides aminés
            modifiable: Liste indiquant quels AA peuvent être modifiés
            d2o_initial: Pourcentage initial de D2O (défaut: 50)
        """
        self.aa_list = aa_list
        self.n_aa = len(aa_list)

        # Si pas de restrictions spécifiées, tous sont modifiables
        if modifiable is None:
            self.modifiable = [True] * self.n_aa
        else:
            assert len(modifiable) == self.n_aa, "modifiable doit avoir la même taille que aa_list, il doit avoir {self.n_aa} éléments"
            self.modifiable = modifiable

        # Vecteur de deutération (True = deutéré, False = non deutéré)
        self.deuteration = [False] * self.n_aa

        # Fitness du chromosome
        self.fitness = 0.0

        self.d2o = self._validate_d2o(d2o_initial)

    @staticmethod
    def _validate_d2o(value: int) -> int:
        """Valide les borne de D2O entre 0 et 100"""
        return max(0, min(int(value), 100))

    def randomize_deuteration(self):
        """
        Initialise aléatoirement le vecteur de deutération (respectant les restrictions)
        """
        for i in range(self.n_aa):
            if self.modifiable[i]:
                self.deuteration[i] = random.choice([True, False])
            else:
                self.deuteration[i] = False

    def modify_d2o(self, variation_rate: int):
        """
        Permet une variation aleatoire du % de D2O
        Args:
             variation_rate: Amplitude de variation possible (5 correspond à ±5%)
        """
        variation = random.randint(-variation_rate, variation_rate)
        self.d2o = self._validate_d2o(self.d2o + variation)


    def get_deuteration_count(self) -> int:
        """Compte le nombre d'AA deutérés"""
        return sum(self.deuteration)

    def copy(self) -> 'Chromosome':
        """Crée une copie du chromosome"""
        new_chrom = Chromosome(self.aa_list, self.modifiable, self.d2o)
        new_chrom.deuteration = self.deuteration.copy()
        new_chrom.fitness = self.fitness
        return new_chrom

    def __str__(self) -> str:
        """Représentation textuelle du chromosome"""
        result = []
        for i, aa in enumerate(self.aa_list):
            if self.deuteration[i]:
                result.append(f"{aa.code_3}(D)")
            else:
                result.append(f"{aa.code_3}(H)")
        return " | ".join(result) + f" | D2O={self.d2o}% | fitness={self.fitness}"

    def to_dict(self) -> dict:
        """Convertit le chromosome en dictionnaire pour sérialisation."""
        return {
            'deuteration': self.deuteration.copy(),
            'd2o': self.d2o,
            'fitness': self.fitness,
            'deuteration_count': self.get_deuteration_count()
        }

# ============================================================================
#                       CLASSE POPULATION GENERATOR
# ============================================================================

class PopulationGenerator:
    """
    Générateur de populations pour l'algorithme génétique.

    Gère la création de populations initiales et l'évolution de populations
    existantes via sélection, croisement et mutation.

    Attributes:
        aa_list: Liste des acides aminés
        modifiable: Restrictions de deutération
        population_size: Taille de la population
        d2o_initial: Pourcentage initial de D2O
        elitism: Nombre d'élites préservés entre générations
    """
    def __init__(self,
                 aa_list: List[AminoAcid],
                 modifiable: List[bool],
                 population_size: int = 100,
                 d2o_initial: int = 50,
                 elitism: int = 2,
                 d2o_variation_rate: int = 5):
        """Initialise le générateur de population."""
        self.aa_list = aa_list
        self.modifiable = modifiable
        self.population_size = population_size
        self.d2o_initial = d2o_initial
        self.elitism = elitism
        self.d2o_variation_rate = d2o_variation_rate

        # Validation
        assert population_size > 0, "population_size doit être > 0"
        assert 0 <= d2o_initial <= 100, "d2o_initial doit être entre 0 et 100"
        assert elitism >= 0, "elitism doit être >= 0"
        assert elitism < population_size, "elitism doit être < population_size"
        assert 0 <= d2o_variation_rate <= 100, "d20_vatiation_rate doit être entre 0 et 100"

    def generate_initial_population(self) -> List[Chromosome]:
        """
        Génère la population initiale (génération 0).

        Crée une population de chromosomes avec deutération aléatoire
        et pourcentage de D2O variant autour d'un D2O initial.

        Returns:
            Liste de chromosomes initialisés aléatoirement
        """
        population = []

        while len(population) < self.population_size:
            # Créer un chromosome
            chrom = Chromosome(self.aa_list, self.modifiable, self.d2o_initial)

            # Initialiser aléatoirement la deutération
            chrom.randomize_deuteration()

            # Initialiser les variation de D2O autour de d20_initial
            chrom.modify_d2o(self.d2o_variation_rate)
            if self._unique_check(chrom, population):
                population.append(chrom)

        return population
    def generate_next_generation(self,
                                previous_population: List[Chromosome],
                                fitness_scores: List[float],
                                mutation_rate: float = 0.1,
                                crossover_rate: float = 0.7,
                                d2o_variation_rate: int = 5) -> List[Chromosome]:
        """
        Génère la prochaine génération à partir de la précédente.

        Applique les opérations génétiques:
            1. Attribution des fitness (programe exterieur)
            2. Sélection par élitisme
            3. Sélection par fitness
            4. Croisement (crossover)
            5. Mutation
            6. Variation du D2O

        Args:
            previous_population: Population de la génération précédente
            fitness_scores: Scores de fitness correspondants
            mutation_rate: Probabilité de mutation par gène (0-1)
            crossover_rate: Probabilité de croisement (0-1)
            d2o_variation_rate: Amplitude max de variation de D2O (ex: 5%)

        Returns:
            Nouvelle population de chromosomes
        """
        # Validation des entrées
        assert len(previous_population) == len(fitness_scores), "Le nombre de fitness doit correspondre à la taille de la population"
        assert 0 <= mutation_rate <= 1, "mutation_rate doit être entre 0 et 1"
        assert 0 <= crossover_rate <= 1, "crossover_rate doit être entre 0 et 1"

        # 1. Attribuer les fitness aux chromosomes
        for chrom, fitness in zip(previous_population, fitness_scores):
            chrom.fitness = fitness

        # 2. Trier par fitness décroissant
        sorted_population = sorted(previous_population,
                                  key=lambda x: x.fitness,
                                  reverse=True)
        # 3. Créer la nouvelle population
        new_population = []

        # Élitisme : conserver les meilleurs individus
        for i in range(self.elitism):
            elite = sorted_population[i].copy()
            new_population.append(elite)

        # 4. Générer le reste de la population
        while len(new_population) < self.population_size:
            # Sélection des parents par probabilité
            parent1, parent2 = self._probability_selection(sorted_population)

            # Croisement
            child1, child2 = self._crossover(parent1, parent2, crossover_rate)

            # Mutation
            self._mutate(child1, mutation_rate)
            self._mutate(child2, mutation_rate)

            # Variation du D2O
            child1.modify_d2o(d2o_variation_rate)
            child2.modify_d2o(d2o_variation_rate)

            # Ajout à la nouvelle population
            if self._unique_check(child1, new_population):
                new_population.append(child1)
                if len(new_population) < self.population_size and self._unique_check(child2, new_population):
                    new_population.append(child2)

        return new_population

    def _unique_check(self,new_chromosome: Chromosome,subpopulation: List[Chromosome]) -> bool:
        """
        Vérification que le chromosome qui va etre ajouter a la pop n'y est pas déjà
        :param new_chromosome: chromosome qu'on veut ajouter
        :param subpopulation: population avec des chromosomes uniques
        :return: False si un chromosome avec les memes motif de deuterisation et do est trouvée, True sinon
        """
        for chromosome in subpopulation:
            if chromosome.deuteration == new_chromosome.deuteration and chromosome.fitness == new_chromosome.fitness:
                return False
        return True

    def _probability_selection(self, population: List[Chromosome]) -> Tuple[Chromosome, Chromosome]:
        """
        Sélection probabiliste basée sur le fitness.

        Sélectionne 2 chromosomes de la population avec une probabilité
        proportionnelle à leur fitness. Plus le fitness est élevé,
        plus la probabilité d'être sélectionné est grande.

        Args:
            population: Population triée par fitness décroissant

        Returns:
            Tuple de deux chromosomes parents sélectionnés
        """
        # Extraire les fitness comme poids
        weights = [chrom.fitness for chrom in population]

        # Sélectionner 2 parents avec probabilité proportionnelle au fitness
        selected = random.choices(population, weights=weights, k=2)

        return selected[0], selected[1]

    def _crossover(self,
                   parent1: Chromosome,
                   parent2: Chromosome,
                   crossover_rate: float) -> Tuple[Chromosome, Chromosome]:
        """
        Opérateur de croisement (crossover) en un point.

        Échange une partie du matériel génétique entre deux parents
        pour créer deux enfants.

        Args:
            parent1: Premier parent
            parent2: Second parent
            crossover_rate: Probabilité d'effectuer le croisement

        Returns:
            Tuple de deux chromosomes enfants
        """
        # Copier les parents
        child1 = parent1.copy()
        child2 = parent2.copy()

        # Appliquer le croisement avec une certaine probabilité
        if random.random() < crossover_rate:
            # Choisir un point de croisement aléatoire
            crossover_point = random.randint(1, len(self.aa_list) - 1)

            # Échanger les gènes après le point de croisement
            for i in range(crossover_point, len(self.aa_list)):
                if self.modifiable[i]:  # Respecter les restrictions
                    child1.deuteration[i] = parent2.deuteration[i]
                    child2.deuteration[i] = parent1.deuteration[i]

        return child1, child2


    def _mutate(self, chromosome: Chromosome, mutation_rate: float):
        """
        Opérateur de mutation.

        Inverse aléatoirement certains gènes du chromosome
        en fonction du taux de mutation. Effectue entre 1 et 3 mutations.

        Args:
            chromosome: Chromosome à muter (modifié en place)
            mutation_rate: Probabilité de mutation par gène
        """
        # Nombre de mutations à effectuer (entre 1 et 3)
        max_mutations = random.randint(1, 3)
        mutation_count = 0

        # Créer une liste des indices modifiables
        modifiable_indices = [i for i in range(len(self.aa_list)) if self.modifiable[i]]

        # Tant que le nombre de mutations n'est pas atteint
        while mutation_count < max_mutations and modifiable_indices:
            # Sélectionner un AA aléatoire parmi les modifiables
            i = random.choice(modifiable_indices)

            # Vérifier si on doit muter ce gène
            if random.random() < mutation_rate:
                # Inverser le gène
                chromosome.deuteration[i] = not chromosome.deuteration[i]
                mutation_count += 1

                # Retirer cet indice pour éviter de le muter deux fois
                modifiable_indices.remove(i)

# Exemple d'utilisation
if __name__ == "__main__":

    # Parser les arguments de ligne de commande
    args = parse_arguments()

    # Fixer la graine aléatoire si spécifiée (pour reproductibilité)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Graine aléatoire fixée à: {args.seed}")

    # Afficher la configuration

    print("CONFIGURATION DE L'ALGORITHME GÉNÉTIQUE")
    print(f"Population size      : {args.population_size}")
    print(f"Élitisme             : {args.elitism}")
    print(f"D2O initial          : {args.d2o_initial}%")
    print(f"D2O variation rate   : ±{args.d2o_variation_rate}%")
    print(f"Mutation rate        : {args.mutation_rate}")
    print(f"Crossover rate       : {args.crossover_rate}")
    print(f"Générations          : {args.generations}")


    # Créer et exécuter l'algorithme génétique
    print("\n>>> GÉNÉRATION 0 - Création de la population")
    generator = PopulationGenerator(
        aa_list=AMINO_ACIDS,
        modifiable=restrictions,
        population_size=args.population_size,
        d2o_initial=args.d2o_initial,
        elitism=args.elitism,
        d2o_variation_rate=args.d2o_variation_rate,
    )

    #Génération et affichage de la population initiale
    pop = generator.generate_initial_population()

    # Simuler des scores de fitness aléatoires (normalement calculés par SANS)
    print("\n>>> Simulation de l'évaluation par SANS...")
    fitness_scores = [random.uniform(0.3, 0.9) for _ in pop]

    for chrom, fitness in zip(pop, fitness_scores):
        chrom.fitness = fitness

    sorted_pop_0 = sorted(pop,
                               key=lambda x: x.fitness,
                               reverse=True)

    for i, chromosome in enumerate(sorted_pop_0):
        print(f"{i+1}. {chromosome}")

    # Génération n : Évolution
    for gen in range(1, args.generations + 1):
        print(f"\n>>> GÉNÉRATION {gen} - Évolution de la population")
        pop = generator.generate_next_generation(
            previous_population=pop,
            fitness_scores=fitness_scores,
            mutation_rate=args.mutation_rate,
            crossover_rate=args.crossover_rate,
            d2o_variation_rate=args.d2o_variation_rate
        )

        # Nouvelle évaluation simulée
        fitness_scores = [random.uniform(0.3, 0.9) for _ in pop]
        for chrom, fitness in zip(pop, fitness_scores):
            chrom.fitness = fitness

        #sorted_pop = sorted(pop,
        #                    key=lambda x: x.fitness,
        #                    reverse=True)
        for i, chromosome in enumerate(pop):
            print(f"{i+1}. {chromosome}")
