import matplotlib.pyplot as plt
import numpy as np
import random
import math

class Individual(object):
    ##Cada individuo tinee un cromosoma con number_genes cantidad de genes
    #En el caso particular de la tarea el cromosoma sería la palabra que
    #representa al individuo y el number_genes la cantidad de caracteres.

    ##le puse alfabeto por si en algun momento necesitamos trabajar con un conjunto distinto.
    ##Ademas le di la posibilidad de recibir un cromosoma ya hecho por nosotros, sino se genera como random.
    def __init__(self, genes_number: int, chromosome: list):
        self.chromosome = chromosome
        self.genes_number = genes_number
    
    def mutate(self, genes_choices: set, mutation_rate: float) -> None:
        if random.random() < mutation_rate:
            gen = random.randint(0, self.genes_number - 1)
            new_gen = [random.choice(genes_choices)]
            chromosome = self.chromosome[:gen] + new_gen + self.chromosome[gen + 1:]
            self.chromosome = chromosome
        return self
    
    def __repr__(self):
        return "".join(self.chromosome)
        

class GA(object):
    # Esta clase va a guardar la poblacion, la solución,
    # Además recibe el alfabeto y la cantidad de genes para crear individuos con estos parametros

    def __init__(self, population_number: int, genes_choices: set, solution, genes_number: int, scoreCalculatorFunction, mutation_rate:int=0.05, success_value=1):
        self.genes_choices = genes_choices
        self.solution = solution
        self.mutation_rate = mutation_rate
        self.genes_number = genes_number
        self.scoreCalculatorFunction = scoreCalculatorFunction
        self.generationNumber = 1
        self.history = []
        self.population_number = population_number
        self.sol_gen = -1
        self.success_value=success_value
        
        self.population = np.array([], dtype=type(Individual))
        for i in range(population_number):
            i = self.generateIndividual()
            self.population = np.append(self.population, i)
        self.append_to_history() # This records the fitness of the first generation
        print(self.population)

        # Checks if the first generation found the solution
        if self.checkSolutionExists():
            self.sol_gen = self.generationNumber
    
    def checkSolutionExists(self):
        return (self.success_value in map(lambda x: self.scoreCalculatorFunction(x, self.solution), self.population) and self.sol_gen==-1)

    def step(self) -> None:
        new_generation = np.array([], dtype=type(Individual))
        for _ in range(self.population_number):
            new_individual = self.crossover(self.selectIndividual(), self.selectIndividual()).mutate(self.genes_choices, self.mutation_rate)
            new_generation = np.append(new_generation, new_individual)
        self.population = new_generation
        self.generationNumber += 1
        
        # Stores the first generation that found the solution
        if self.checkSolutionExists():
            self.sol_gen=self.generationNumber
            
        # Stores the current status of the Genetic Algorithm
        self.append_to_history()

    def append_to_history(self) -> None:
        self.history.append(
            (
                self.generationNumber,
                self.calculateTotalFitness(),
                self.totalSuccess()
            )
        )
    
    def totalSuccess(self) -> int:
        l = list(self.calculatePopulationFitness())
        return l.count(self.success_value)
    
    def generateIndividual(self) -> Individual:
        """
        Generates a new Individual.
        """
        chromosome: list = []
        for _ in range(self.genes_number):
            chromosome += [random.choice(self.genes_choices)]
        return Individual(self.genes_number, chromosome)

    def selectIndividual(self) -> Individual:
        """
        TOURNAMENT
        Choose 10 random individuals from the population, and selects the individual
        with the best fitness.
        """
        max_score = -1
        individual = None
        for _ in range(10):
            i = random.choice(self.population)
            score = self.scoreCalculatorFunction(i, self.solution)
            if score > max_score:
                individual = i
                max_score = score
        return individual

    def crossover(self, i1: Individual, i2: Individual) -> Individual:
        """
        Generates a new Individual from two parents, taking and mixing the half genes of each parent.
        """
        mid_gen = random.randint(0, len(i1.chromosome)) # we assume the length of both chromosomes are equal
        new_i = Individual(self.genes_number, chromosome = i1.chromosome[:mid_gen] + i2.chromosome[mid_gen:])
        return new_i

    ##hice que esta funcion calcule el score de cada individuo y lo guarde en un arreglo
    ##en select individual solo enremos que llamar a esta funcion y saber en qué posicion esta el maximo
    ##luego se saca el individuo que está en esa misma posicion en el arreglo self.population
    def calculatePopulationFitness(self):
        p_fitness= np.array([], dtype=int)
        for individual in self.population:
            i_score = self.scoreCalculatorFunction(individual, self.solution)
            p_fitness = np.append(p_fitness, i_score)
        return p_fitness
    
    def calculateTotalFitness(self):
        return sum(self.calculatePopulationFitness())

def WORD_FITNESS(individual, solution):
    score=0
    if individual.chromosome == solution:
        return len(solution)
    for i in range(individual.genes_number):
        if individual.chromosome[i] == solution[i]:
            score += 1
    score = score/len(solution) # between 0 and 1
    return score

def BINARY_FITNESS(individual, solution):
    decimal = 0
    for i in range(individual.genes_number):
        decimal += (2**i)*int(individual.chromosome[::-1][i])
    score = (solution - abs(solution - decimal))/solution # between 0 and 1
    return score

#GENES_CHOICES = "abcdefghijklmnopqrstuvwxyz., "
#SOLUTION = list("lorem")
#GENES_NUMBER = len(SOLUTION)
#SCORE_FUNCTION = WORD_FITNESS

GENES_CHOICES = list("01")
SOLUTION = 1000000
GENES_NUMBER = int(math.log(SOLUTION,2)) + 1
SCORE_FUNCTION = BINARY_FITNESS

POPULATION_NUMBER = 150
MUTATION_RATE = 0.25

ga = GA(population_number=POPULATION_NUMBER,
        genes_choices=GENES_CHOICES,
        solution=SOLUTION,
        genes_number=GENES_NUMBER,
        scoreCalculatorFunction=SCORE_FUNCTION,
        mutation_rate=MUTATION_RATE)

##################################################

NUMBER_OF_GENERATIONS = 150

for i in range(NUMBER_OF_GENERATIONS):
    ga.step()
print(ga.history)
print(ga.population)


# CHART: TOTAL SCORE OF EVERY GENERATION
GENERATIONS = []
TOTAL_SCORES = []
ACIERTOS=[]
for i in ga.history:
    GENERATIONS.append(i[0])
    TOTAL_SCORES.append(i[1])
    ACIERTOS.append(i[2])
    
print("Solution found after:", ga.sol_gen, "generations\n")

plt.grid(zorder=0)

plt.subplot(1,2,1)
plt.title("Score por generación")
plt.ylabel("Cantidad total de score")
plt.xlabel("Número de generación")
plt.plot(GENERATIONS, TOTAL_SCORES, color="blue", label="Total Score")

plt.subplot(1,2,2)
plt.title("Aciertos por generación")
plt.ylabel("Cantidad de aciertos")
plt.xlabel("Número de generación")
plt.plot(GENERATIONS, ACIERTOS, color="green", label="Total Successes")

plt.show()
plt.close()
