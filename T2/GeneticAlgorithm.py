import numpy as np
import random

import abc
from typing import List, Any

class AbstractIndividual(abc.ABC):
    # Clase Abstracta que representa a un individuo mediante una lista
    # de genes que representa <i>chromosome</i> y posee una cantidad
    # <i>genes_number</i> de elementos (genes).

    def __init__(self, genes_number: int, chromosome: list):
        self.chromosome = chromosome
        self.genes_number = genes_number
        self.fitness = None
    
    def mutate(self, genes_choices, mutation_rate: float) -> None:
        # Permite reemplazar uno de sus genes aleatoriamente por otro presente
        # dentro de los <i>genes_choices</i>.
        if random.random() < mutation_rate:
            gen = random.randint(0, self.genes_number - 1)
            new_gen = [random.choice(genes_choices)]
            chromosome = self.chromosome[:gen] + new_gen + self.chromosome[gen + 1:]
            self.chromosome = chromosome
        return self

    def __repr__(self):
        return " | ".join(map(str, self.chromosome))

class Individual(AbstractIndividual):
    # Clase estricta de un individuo, permite instanciar de <i>AbstractIndividual</i>
    # tal como está.
    def __init__(self, genes_number: int, chromosome: list):
        super().__init__(genes_number, chromosome)

class GA(object):
    # Clase que permite configurar un Algoritmo Genético.
    #   population_number <int>: El número de individuos presentes en cada generación.
    #   genes_choices <List[Any]>: Lista de genes posibles para un individuo.
    #   solution <Any>: La solución buscada para el problema, notar que puede ser de cualquier tipo.
    #   genes_number <int>: Cantidad de genes presentes en el cromosoma de cada invidiuo.
    #   scoreCalculatorFunction <function(<AbstractIndividual>, <Any>, <dict>)>: Función que permite calcular el fitness
    # de un individuo en particular, dada la solución y los argumentos extra definidos en un diccionario.
    #   mutation_rate <float>: Define la probabilidad de mutación luego del cruce de individuos.
    #   success_value <float>: Define el valor para el cual un individuo es considerado solución.
    # es necesario mejorar este parámetro para considerar una función validadora de soluciones.
    #   options <dict>: Contiene parámetros extra dependiendo del tipo de problema a resolver.

    def __init__(self, population_number: int, genes_choices: List[Any], solution: Any, genes_number: int,
                 scoreCalculatorFunction, mutation_rate: float = 0.05, success_value: float = 1.0, options: dict = {}):
        self.genes_choices = genes_choices
        self.solution = solution
        self.mutation_rate = mutation_rate
        self.genes_number = genes_number
        self.scoreCalculatorFunction = \
            lambda individual, solution, options: individual.fitness or \
                                                  scoreCalculatorFunction(individual, solution, options)
        self.generationNumber = 1
        self.history = []
        self.population_number = population_number
        self.sol_gen = -1
        self.success_value=success_value
        self.scorePerIndividual=[]
        self.options = options

        self.IndividualClass = options.get("IndividualClass") or Individual
        
        self.population = np.array([], dtype=type(AbstractIndividual))
        for i in range(population_number):
            i = self.generateIndividual()
            self.population = np.append(self.population, i)
        self.append_to_history() # This records the fitness of the first generation

        # Checks if the first generation found the solution
        if self.checkSolutionExists():
            self.sol_gen = self.generationNumber
    
    def checkSolutionExists(self):
        # Checks if some individual is the solution of the problem, compairing its score with the success_value.
        return (self.success_value in self.scorePerIndividual and self.sol_gen == -1)

    def step(self) -> None:
        # Steps one generation.
        # Uses tournament selection.
        new_generation = np.array([], dtype=type(AbstractIndividual))
        for _ in range(self.population_number):
            new_individual = self.crossover(self.selectIndividual(), self.selectIndividual()).mutate(self.genes_choices, self.mutation_rate)
            new_generation = np.append(new_generation, new_individual)
        self.population = new_generation
        self.generationNumber += 1
        
        # Checks if this new generation was the first one that found the solution.
        if self.checkSolutionExists():
            self.sol_gen=self.generationNumber
            
        # Stores the current status of the Genetic Algorithm.
        self.append_to_history()

    def append_to_history(self) -> None:
        # Adds to the history of the algorithm, data of the current generation.
        # Currently stores: The number of generation,
        #                   The total fitness
        #                   The total successes
        self.history.append(
            (
                self.generationNumber,
                self.calculateTotalFitness() / self.population_number * 100,
                self.totalSuccess()
            )
        )
    
    def totalSuccess(self) -> int:
        # Returns the current total succeses as percentage (Values between 0 and 1).
        l = list(self.calculatePopulationFitness())
        return l.count(self.success_value) / self.population_number * 100
    
    def generateIndividual(self) -> AbstractIndividual:
        # Generates a new Individual.
        chromosome: list = []
        for _ in range(self.genes_number):
            chromosome += [random.choice(self.genes_choices)]
        return self.IndividualClass(self.genes_number, chromosome)

    def selectIndividual(self) -> AbstractIndividual:
        # TOURNAMENT SELECTION
        # Choose 10 random individuals from the population, and selects the individual
        # with the best fitness.
        max_score = -1
        individual = None
        for _ in range(10):
            i = random.choice(self.population)
            score = self.scoreCalculatorFunction(i, self.solution, options=self.options)
            if score > max_score:
                individual = i
                max_score = score
        return individual

    def crossover(self, i1: AbstractIndividual, i2: AbstractIndividual) -> AbstractIndividual:
        # Generates a new Individual from two parents, taking and mixing the half genes of each parent.
        mid_gen = random.randint(0, len(i1.chromosome)) # we assume the length of both chromosomes are equal
        new_i = self.IndividualClass(self.genes_number, chromosome = i1.chromosome[:mid_gen] + i2.chromosome[mid_gen:])
        return new_i

    # Hice que esta funcion calcule el score de cada individuo y lo guarde en un arreglo
    # en select individual solo enremos que llamar a esta funcion y saber en qué posicion esta el maximo
    # luego se saca el individuo que está en esa misma posicion en el arreglo self.population.
    def calculatePopulationFitness(self):
        p_fitness= []
        for individual in self.population:
            i_score = self.scoreCalculatorFunction(individual, self.solution, options=self.options)
            p_fitness.append(i_score)
        self.scorePerIndividual = p_fitness
        return p_fitness
    
    def calculateTotalFitness(self):
        # Returns the sum of the fitness of every individual in the current generation.
        return sum(self.calculatePopulationFitness())

    def showSolution(self):
        # Prints the first ocurrence of the individual that is considered as a solution.
        try:
            elementIndex = self.scorePerIndividual.index(1)
            print(self.population[elementIndex])
        except ValueError:
            print("No existe solución en esta generación.")

    def stadistics(self):
        # Returns a 3-tuple containing a list of the statistics of this genetic algorithm.
        # Very useful for graphs.
        GENERATIONS = []
        TOTAL_SCORES = []
        SUCCESS=[]
        for i in self.history:
            GENERATIONS.append(i[0])
            TOTAL_SCORES.append(i[1])
            SUCCESS.append(i[2])
        return GENERATIONS, TOTAL_SCORES, SUCCESS
