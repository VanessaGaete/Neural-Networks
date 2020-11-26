import matplotlib.pyplot as plt
import numpy as np
import random
import math

class Board(object):
    def __init__(self, dim: int):
        self.dim = dim
        self.board = np.zeros((dim,dim))
        self.queens = 0

    def countCollisions(self) -> int:
        # Counts the amount of collisions between the queens of th board
        collisions = 0
        for x in range(self.dim):
            for y in range(self.dim):
                if self.isOcuppied(x, y):
                    collisions += self.countAttacks(x,y)
        return collisions//2

    def addQueen(self, x: int, y: int) -> bool:
        # Adds a queen in the given coordinates
        if not self.isOcuppied(x, y):
            self.board[x,y] = 1
            self.queens += 1
            return True
        return False

    def removeQueen(self, x: int, y: int) -> bool:
        # Removes a queen in the given coordinates
        if self.isOcuppied(x, y):
            self.board[x,y] = 0
            self.queens -= 1
            return True
        return False

    def clean(self) -> None:
        # Cleans the board, setting a new one
        self.board = np.zeros((self.dim, self.dim))
        self.queens = 0

    def isOcuppied(self, x: int, y: int) -> bool:
        # Checks if a box contains a queen
        return bool(self.board[x,y])

    def countAttacks(self, x: int, y: int) -> bool:
        # Returns True if this box can be attacked
        # by another queen on the board
        counter = 0
        counter += self.countRowAttacks(x, y)
        counter += self.countColumnAttacks(x, y)

        counter += self.countDiagonalAttacks(x-1, y-1, -1, -1) + \
            self.countDiagonalAttacks(x+1, y+1, self.dim, self.dim) + \
            self.countDiagonalAttacks(x-1, y+1, -1, self.dim) + \
            self.countDiagonalAttacks(x+1, y-1, self.dim, -1)

        return counter
        
    def countRowAttacks(self, x:int, y:int) -> bool:
        # Checks a row, returns True if there is a queen on that row
        count=0
        for i in range(0, y):
            if self.isOcuppied(x, i):
                count += 1
        for i in range(y+1, self.dim):
            if self.isOcuppied(x, i):
                count += 1
        return count
    
    def countColumnAttacks(self, x: int, y: int) -> bool:
        # Checks a column, returns True if there is a queen on that column
        count = 0
        for i in range(0, x):
            if self.isOcuppied(i, y):
                count += 1
        for i in range(x+1, self.dim):
            if self.isOcuppied(i, y):
                count += 1
        return count
        
    def countDiagonalAttacks(self, x: int, y: int, finalx: int, finaly: int) -> bool:
        # Checks a diagonal, returns True if there is a queen on that diagonal+
        count = 0
        if (x < 0 or x >= self.dim or y < 0 or y >= self.dim):
            return 0
        while x!=finalx and y!=finaly:
            if self.isOcuppied(x,y):
                count+=1
            if finalx==-1:
                x-=1
            else:
                x+=1
            if finaly==-1:
                y-=1
            else:
                y+=1
        return count

    def isValid(self) -> bool:
        # Checks if all the queens are set in the board and checks if none
        # of the queens of the board are attacking to another one.
        if self.queens != self.dim:
            return False
            
        for x in range(self.dim):
            for y in range(self.dim):
                if self.isOcuppied(x,y):
                    if self.countAttacks(x, y) != 0:
                        return False
        return True
    
    def printBoard(self):
        print(self.board)
        print("Collisions:", self.countCollisions())
        print("Valid?:", self.isValid())


class AbstractIndividual(object):
    #Cada individuo tinee un cromosoma con number_genes cantidad de genes
    #En el caso particular de la tarea el cromosoma sería la palabra que
    #representa al individuo y el number_genes la cantidad de caracteres.

    ##le puse alfabeto por si en algun momento necesitamos trabajar con un conjunto distinto.
    ##Ademas le di la posibilidad de recibir un cromosoma ya hecho por nosotros, sino se genera como random.
    def __init__(self, genes_number: int, chromosome: list):
        self.chromosome = chromosome
        self.genes_number = genes_number
        self.fitness = None
    
    def mutate(self, genes_choices, mutation_rate: float) -> None:
        if random.random() < mutation_rate:
            gen = random.randint(0, self.genes_number - 1)
            new_gen = [random.choice(genes_choices)]
            chromosome = self.chromosome[:gen] + new_gen + self.chromosome[gen + 1:]
            self.chromosome = chromosome
        return self

    def __repr__(self):
        return " | ".join(map(str, self.chromosome))

#class Individual(AbstractIndividual):
#    def __init__(self, genes_number: int, chromosome: list):
#        super().__init__(genes_number, chromosome)

class Individual(AbstractIndividual):
    def __init__(self, genes_number: int, chromosome: list):
        super().__init__(genes_number, chromosome)
        self.board=[]

    def createBoard(self, nQueens, dimension):
        board = Board(dimension)
        for i in range(0, nQueens*2, 2):
            x = self.chromosome[i]
            y = self.chromosome[i+1]
            board.addQueen(x,y)
        self.board=board
    
    def __repr__(self):
        return str(self.board.board)

class GA(object):
    # Esta clase va a guardar la poblacion, la solución,
    # Además recibe el alfabeto y la cantidad de genes para crear individuos con estos parametros

    def __init__(self, population_number: int, genes_choices, solution, genes_number: int, scoreCalculatorFunction, mutation_rate:int=0.05, success_value=1, **kwargs):
        self.genes_choices = genes_choices
        self.solution = solution
        self.mutation_rate = mutation_rate
        self.genes_number = genes_number
        self.scoreCalculatorFunction = lambda individual, solution, kwargs: individual.fitness or scoreCalculatorFunction(individual, solution, kwargs)
        self.generationNumber = 1
        self.history = []
        self.population_number = population_number
        self.sol_gen = -1
        self.success_value=success_value
        self.scorePerIndividual=[]
        self.individual_kwargs = kwargs
        
        self.population = np.array([], dtype=type(Individual))
        for i in range(population_number):
            i = self.generateIndividual()
            self.population = np.append(self.population, i)
        self.append_to_history() # This records the fitness of the first generation

        # Checks if the first generation found the solution
        if self.checkSolutionExists():
            self.sol_gen = self.generationNumber
    
    def checkSolutionExists(self):
        return (self.success_value in self.scorePerIndividual and self.sol_gen==-1)

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
                self.calculateTotalFitness() / self.population_number * 100,
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
            score = self.scoreCalculatorFunction(i, self.solution, kwargs=self.individual_kwargs)
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
        p_fitness= []
        for individual in self.population:
            i_score = self.scoreCalculatorFunction(individual, self.solution, kwargs=self.individual_kwargs)
            p_fitness.append(i_score)
        self.scorePerIndividual = p_fitness
        return p_fitness
    
    def calculateTotalFitness(self):
        return sum(self.calculatePopulationFitness())

    def showSolution(self):
        try:
            elementIndex = self.scorePerIndividual.index(1)
            print(self.population[elementIndex])
        except ValueError:
            print("No existe solución en esta generación.")

############################################################

def WORD_FITNESS(individual, solution, kwargs):
    score=0
    if individual.chromosome == solution:
        return len(solution)
    for i in range(individual.genes_number):
        if individual.chromosome[i] == solution[i]:
            score += 1
    score = score/len(solution) # between 0 and 1
    return score

def BINARY_FITNESS(individual, solution, kwargs):
    decimal = 0
    for i in range(individual.genes_number):
        decimal += (2**i)*int(individual.chromosome[::-1][i])
    score = (solution - abs(solution - decimal))/solution # between 0 and 1
    return score

def N_QUEEN_FITNESS(individual, solution, kwargs):
    nQueens = solution
    dimension = kwargs["dimension"]
    
    individual.createBoard(nQueens, dimension)
    fitness = (solution - individual.board.countCollisions()) * (individual.board.queens) / (solution**2)
    individual.fitness = fitness
    return fitness

#SOLUTION = list("lorem")
#GENES_CHOICES = "abcdefghijklmnopqrstuvwxyz., "
#GENES_NUMBER = len(SOLUTION)
#SCORE_FUNCTION = WORD_FITNESS

#SOLUTION = 1000000
#GENES_CHOICES = list("01")
#GENES_NUMBER = int(math.log(SOLUTION,2)) + 1
#SCORE_FUNCTION = BINARY_FITNESS

SOLUTION = 4 # N QUEENS
DIMENSION = 4
GENES_CHOICES = list(range(DIMENSION))
GENES_NUMBER = SOLUTION*2 # (x,y) sequence, 2 by 2
SCORE_FUNCTION = N_QUEEN_FITNESS

POPULATION_NUMBER = 75
MUTATION_RATE = 0.25

ga = GA(population_number=POPULATION_NUMBER,
        genes_choices=GENES_CHOICES,
        solution=SOLUTION,
        genes_number=GENES_NUMBER,
        scoreCalculatorFunction=SCORE_FUNCTION,
        mutation_rate=MUTATION_RATE,
        dimension=DIMENSION)

##################################################

NUMBER_OF_GENERATIONS = 100

for i in range(NUMBER_OF_GENERATIONS):
    ga.step()

print(ga.history)
ga.showSolution()


# CHART: TOTAL SCORE OF EVERY GENERATION
GENERATIONS = []
TOTAL_SCORES = []
ACIERTOS=[]
for i in ga.history:
    GENERATIONS.append(i[0])
    TOTAL_SCORES.append(i[1])
    ACIERTOS.append(i[2])
    
print("Solution found after:", ga.sol_gen, "generations\n")

plt.subplot(1,2,1)
plt.title("Score por generación")
plt.ylabel("% Score total poblacional")
plt.xlabel("# Generación")
plt.plot(GENERATIONS, TOTAL_SCORES, color="blue", label="Total Score")
plt.grid()

plt.subplot(1,2,2)
plt.title("Aciertos por generación")
plt.ylabel("# Aciertos")
plt.xlabel("# Generación")
plt.plot(GENERATIONS, ACIERTOS, color="green", label="Total Successes")
plt.grid()

plt.show()
plt.close()
