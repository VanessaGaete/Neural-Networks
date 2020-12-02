import math

from GeneticAlgorithm import AbstractIndividual, report
import ReportUtils

class Individual(AbstractIndividual):
    # Clase estricta de un individuo, permite instanciar de <i>AbstractIndividual</i>
    # tal como está.
    def __init__(self, genes_number: int, chromosome: list):
        super().__init__(genes_number, chromosome)

def BINARY_FITNESS(individual, solution, options):
    decimal = 0
    for i in range(individual.genes_number):
        decimal += (2**i)*int(individual.chromosome[::-1][i])
    score = (solution - abs(solution - decimal))/solution # between 0 and 1
    return score

SOLUTION = 300
GENES_CHOICES = list("01")
GENES_NUMBER = int(math.log(SOLUTION,2)) + 1
SCORE_FUNCTION = BINARY_FITNESS
OPTIONS = {}

POPULATION_NUMBER = 100
MUTATION_RATE = 0.25
NUMBER_OF_GENERATIONS = 100

###########################################ESTUDIO DE HIPERPARAMETROS#########################################

# Probar el algoritmo con los parámetros dados / 1 reporte
report(SOLUTION, GENES_CHOICES, GENES_NUMBER, SCORE_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS, OPTIONS)

# Probar el algoritmo con los parámetros dados y combinaciones de otros / 7 reportes + gráficos
# ReportUtils.main(SOLUTION, GENES_CHOICES, GENES_NUMBER, SCORE_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS)

###########################################ESTUDIO DE DISTINTOS VALORES PARA EL PROBLEMA#########################################

# Probar distintas soluciones simultáneamente y graficar los score y aciertos para ellas

# import matplotlib.pyplot as plt

# SOLUTION = 10
# GENES_NUMBER = int(math.log(SOLUTION,2)) + 1

# generations1, total_score1, success1=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,20,MUTATION_RATE,NUMBER_OF_GENERATIONS)

# SOLUTION = 100
# GENES_NUMBER = int(math.log(SOLUTION,2)) + 1

# generations2, total_score2, success2=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,50,MUTATION_RATE,NUMBER_OF_GENERATIONS)

# SOLUTION = 500
# GENES_NUMBER = int(math.log(SOLUTION,2)) + 1

# generations3, total_score3, success3=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,50,MUTATION_RATE,NUMBER_OF_GENERATIONS)

# SOLUTION = 1000
# GENES_NUMBER = int(math.log(SOLUTION,2)) + 1

# generations4, total_score4, success4=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,75,MUTATION_RATE,NUMBER_OF_GENERATIONS)


# f1 = plt.figure(1)
# ax1 = f1.add_subplot(111)
# ax1.set_title("Score total poblacional (Mutation rate: %s%%)" % (MUTATION_RATE*100))
# ax1.set_xlabel("# Generación")
# ax1.set_ylabel("% Score total poblacional")
# ax1.plot(generations1,total_score1, color="blue", label="número a convertir: 10")
# ax1.plot(generations2,total_score2, color="red", label="número a convertir: 100")
# ax1.plot(generations3,total_score3, color="y", label="número a convertir: 500")
# ax1.plot(generations4,total_score4, color='g', label="número a convertir: 1000")
# plt.legend()
# plt.grid()
# plt.yticks(range(0, 101, 20))
# f1.show()

# f2 = plt.figure(2)
# ax2 = f2.add_subplot(111)
# ax2.set_title("Aciertos por generación (Mutation rate: %s%%)" % (MUTATION_RATE*100))
# ax2.set_xlabel("# Generación")
# ax2.set_ylabel("% Aciertos")
# ax2.plot(generations1,success1, color="blue", label="número a convertir: 10")
# ax2.plot(generations2,success2, color="red", label="número a convertir: 100")
# ax2.plot(generations3,success3, color="y", label="número a convertir: 500")
# ax2.plot(generations4,success4, color='g', label="número a convertir: 1000")
# plt.legend()
# plt.grid()
# plt.yticks(range(0, 101, 20))
# f2.show()

# plt.show()
# plt.close()