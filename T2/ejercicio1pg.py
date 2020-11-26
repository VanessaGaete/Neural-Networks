import matplotlib.pyplot as plt
import numpy as np
import random

class Individual(object):
    ##Cada individuo tinee un cromosoma con number_genes cantidad de genes
    #En el caso particular de la tarea el cromosoma sería la palabra que
    #representa al individuo y el number_genes la cantidad de caracteres.

    ##le puse alfabeto por si en algun momento necesitamos trabajar con un conjunto distinto.
    ##Ademas le di la posibilidad de recibir un cromosoma ya hecho por nosotros, sino se genera como random.
    def __init__(self, genes_number, chromosome):
        self.chromosome = chromosome
        self.genes_number = genes_number

    def calculateIndividualScore(self, solution):
        score=0
        if self.chromosome == solution:
            return len(solution)
        for i in range(self.genes_number):
            if self.chromosome[i] == solution[i]:
                score += 1
        return score
    
    def mutate(self, genes_choices, mutation_rate: float) -> None:
        if random.random() < mutation_rate:
            gen = random.randint(0, self.genes_number - 1)
            new_gen = random.choice(genes_choices)
            chromosome = self.chromosome[:gen] + new_gen + self.chromosome[gen + 1:]
            self.chromosome = chromosome
        return self
    
    def __repr__(self):
        return self.chromosome
        

class GA(object):
    # Esta clase va a guardar la poblacion, la solución,
    # Además recibe el alfabeto y la cantidad de genes para crear individuos con estos parametros

    def __init__(self, population_number, alphabet, solution, genes_number, mutation_rate=0.05):
        self.alphabet = alphabet
        self.solution = solution
        self.mutation_rate = mutation_rate
        self.genes_number = genes_number
        self.generationNumber = 1
        self.history = []
        self.population_number = population_number
        self.sol_gen=0
        
        self.population = np.array([], dtype=type(Individual))
        for i in range(population_number):
            i = self.generateIndividual()
            self.population = np.append(self.population, i)
        self.append_to_history() # This records the fitness of the first generation
        print(self.population)

    def step(self):
        """
        1.- 
        """
        new_generation = np.array([], dtype=type(Individual))
        for _ in range(self.population_number):
            new_individual = self.crossover(self.selectIndividual(), self.selectIndividual()).mutate(self.alphabet, self.mutation_rate)
            new_generation = np.append(new_generation, new_individual)
        self.population = new_generation
        self.generationNumber += 1
        
        if (self.solution in map(lambda x: x.chromosome, new_generation)) == True and self.sol_gen==0:
            self.sol_gen=self.generationNumber
            
        self.append_to_history()

    def append_to_history(self):
        self.history.append(
            (
                self.generationNumber,
                self.calculateTotalFitness(),
                self.totalSuccess()
            )
        )
    
    def totalSuccess(self):
        l = list(self.calculatePopulationFitness())
        return l.count(len(FINAL_WORD))
    
    def generateIndividual(self) -> Individual:
        """
        Generates a new Individual.
        """
        chromosome = ''
        for i in range(self.genes_number):
            chromosome += random.choice(self.alphabet)
        return Individual(self.genes_number, chromosome)

    def selectIndividual(self) -> Individual:
        """
        TOURNAMENT
        Choose 10 random individuals from the population, and selects the individual
        with the best fitness.
        """
        max_score = -1
        individual = None
        for n in range(10):
            i = random.choice(self.population)
            score = i.calculateIndividualScore(self.solution)
            if score > max_score:
                individual = i
                max_score = score
        return individual

    def crossover(self, i1: Individual, i2: Individual) -> Individual:
        """
        Generates a new Individual from two parents, taking and mixing the half genes of each parent.
        """
        mid_gen = random.randint(0, len(self.solution))
        new_i = Individual(self.genes_number, chromosome = i1.chromosome[:mid_gen] + i2.chromosome[mid_gen:])
        return new_i

    ##hice que esta funcion calcule el score de cada individuo y lo guarde en un arreglo
    ##en select individual solo enremos que llamar a esta funcion y saber en qué posicion esta el maximo
    ##luego se saca el individuo que está en esa misma posicion en el arreglo self.population
    def calculatePopulationFitness(self):
        p_fitness= np.array([], dtype=int)
        for individual in self.population:
            i_score=individual.calculateIndividualScore(self.solution)
            p_fitness = np.append(p_fitness, i_score)
        return p_fitness
    
    def calculateTotalFitness(self):
        return sum(self.calculatePopulationFitness())

POPULATION_NUMBER = 150
ALPHABET = "abcdefghijklmnopqrstuvwxyz., "
FINAL_WORD = "lorem"
MUTATION_RATE = 0.1

ga = GA(POPULATION_NUMBER, ALPHABET, FINAL_WORD, len(FINAL_WORD), MUTATION_RATE)

NUMBER_OF_GENERATIONS = 50

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
