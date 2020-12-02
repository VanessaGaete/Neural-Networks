import numpy as np

from GeneticAlgorithm import AbstractIndividual, report
import ReportUtils

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

def N_QUEEN_FITNESS(individual, solution, options):
    nQueens = solution
    dimension = options["dimension"]
    
    individual.createBoard(nQueens, dimension)
    fitness = (solution - individual.board.countCollisions()) * (individual.board.queens) / (solution**2)
    individual.fitness = fitness
    return fitness

SOLUTION = 4 # N queens
DIMENSION = 4 # MxM board
GENES_CHOICES = list(range(DIMENSION)) # [0...M]
GENES_NUMBER = SOLUTION*2 # (x,y) sequence, 2 by 2
SCORE_FUNCTION = N_QUEEN_FITNESS
OPTIONS = {"dimension": DIMENSION,
           "IndividualClass": Individual}

POPULATION_NUMBER = 75
MUTATION_RATE = 0.25
NUMBER_OF_GENERATIONS = 100

###########################################ESTUDIO DE HIPERPARAMETROS#########################################

# Probar el algoritmo con los parámetros dados / 1 reporte
report(SOLUTION, GENES_CHOICES, GENES_NUMBER, SCORE_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS, OPTIONS)

# Probar el algoritmo con los parámetros dados y combinaciones de otros / 7 reportes + gráficos
# ReportUtils.main(SOLUTION, GENES_CHOICES, GENES_NUMBER, SCORE_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS, OPTIONS)

###########################################ESTUDIO DE DISTINTOS VALORES PARA EL PROBLEMA#########################################

# Probar distintas soluciones simultáneamente y graficar los score y aciertos para ellas

# from GeneticAlgorithm import report
# import matplotlib.pyplot as plt

# SOLUTION = 5 # N queens
# DIMENSION = 5 # MxM board
# GENES_CHOICES = list(range(DIMENSION)) # [0...M]
# GENES_NUMBER = SOLUTION*2 # (x,y) sequence, 2 by 2
# OPTIONS = {"dimension": DIMENSION,
#            "IndividualClass": Individual}

# generations1, total_score1, success1=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,20,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)

# SOLUTION = 8 # N queens
# DIMENSION = 8 # MxM board
# GENES_CHOICES = list(range(DIMENSION)) # [0...M]
# GENES_NUMBER = SOLUTION*2 # (x,y) sequence, 2 by 2
# OPTIONS = {"dimension": DIMENSION,
#            "IndividualClass": Individual}

# generations2, total_score2, success2=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,50,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)

# SOLUTION = 4 # N queens
# DIMENSION = 8 # MxM board
# GENES_CHOICES = list(range(DIMENSION)) # [0...M]
# GENES_NUMBER = SOLUTION*2 # (x,y) sequence, 2 by 2
# OPTIONS = {"dimension": DIMENSION,
#            "IndividualClass": Individual}
# generations3, total_score3, success3=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,100,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)

# f1 = plt.figure(1)
# ax1 = f1.add_subplot(111)
# ax1.set_title("Score total poblacional (Mutation rate: %s%%)" % (MUTATION_RATE*100))
# ax1.set_xlabel("# Generación")
# ax1.set_ylabel("% Score total poblacional")
# ax1.plot(generations1,total_score1, color="blue", label="5 reinas en 5x5")
# ax1.plot(generations2,total_score2, color="red", label="8 reinas en 8x8")
# ax1.plot(generations3,total_score3, color='g', label="4 reinas en 8x8")
# plt.legend()
# plt.grid()
# plt.yticks(range(0, 101, 20))
# f1.show()

# f2 = plt.figure(2)
# ax2 = f2.add_subplot(111)
# ax2.set_title("Aciertos por generación (Mutation rate: %s%%)" % (MUTATION_RATE*100))
# ax2.set_xlabel("# Generación")
# ax2.set_ylabel("% Aciertos")
# ax2.plot(generations1,success1, color="blue", label="5 reinas en 5x5")
# ax2.plot(generations2,success2, color="red", label="8 reinas en 8x8")
# ax2.plot(generations3,success3, color='g', label="4 reinas en 8x8")
# plt.legend()
# plt.grid()
# plt.yticks(range(0, 101, 20))
# f2.show()

# plt.show()
# plt.close()