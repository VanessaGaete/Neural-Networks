from GeneticAlgorithm import AbstractIndividual, report
import ReportUtils

class Individual(AbstractIndividual):
    # Clase estricta de un individuo, permite instanciar de <i>AbstractIndividual</i>
    # tal como está.
    def __init__(self, genes_number: int, chromosome: list):
        super().__init__(genes_number, chromosome)

def WORD_FITNESS(individual, solution, options):
    score=0
    for i in range(individual.genes_number):
        if individual.chromosome[i] == solution[i]:
            score += 1
    score = score/len(solution) # between 0 and 1
    return score

SOLUTION = list("lorem")
GENES_CHOICES = list("abcdefghijklmnopqrstuvwxyz., ")
GENES_NUMBER = len(SOLUTION)
SCORE_FUNCTION = WORD_FITNESS

POPULATION_NUMBER = 75
MUTATION_RATE = 0.25
NUMBER_OF_GENERATIONS = 100

###########################################ESTUDIO DE HIPERPARAMETROS#########################################
ReportUtils.main(SOLUTION, GENES_CHOICES, GENES_NUMBER, SCORE_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS)

###########################################ESTUDIO DE DISTINTOS VALORES PARA EL PROBLEMA#########################################
# from GeneticAlgorithm import report
# import matplotlib.pyplot as plt

# SOLUTION = list("car")
# GENES_NUMBER = len(SOLUTION)

# generations1, total_score1, success1=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,20,MUTATION_RATE,NUMBER_OF_GENERATIONS)

# SOLUTION = list("Lorem ipsum dolor sit amet")
# GENES_NUMBER = len(SOLUTION)

# generations2, total_score2, success2=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,50,MUTATION_RATE,NUMBER_OF_GENERATIONS)

# SOLUTION = list("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua.")
# GENES_NUMBER = len(SOLUTION)

# generations3, total_score3, success3=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,50,MUTATION_RATE,NUMBER_OF_GENERATIONS)

# SOLUTION = list("Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed eiusmod tempor incidunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquid ex ea commodi consequat. Quis aute iure reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.")
# GENES_NUMBER = len(SOLUTION)

# generations4, total_score4, success4=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,75,MUTATION_RATE,NUMBER_OF_GENERATIONS)


# f1 = plt.figure(1)
# ax1 = f1.add_subplot(111)
# ax1.set_title("Score total poblacional (Mutation rate: %s%%)" % (MUTATION_RATE*100))
# ax1.set_xlabel("# Generación")
# ax1.set_ylabel("% Score total poblacional")
# ax1.plot(generations1,total_score1, color="blue", label="Palabra de largo 3")
# ax1.plot(generations2,total_score2, color="red", label="Palabra de largo 26")
# ax1.plot(generations3,total_score3, color="y", label="Palabra de largo 118")
# ax1.plot(generations4,total_score4, color='g', label="Palabra de largo 309")
# plt.legend()
# plt.grid()
# plt.yticks(range(0, 101, 20))
# f1.show()

# f2 = plt.figure(2)
# ax2 = f2.add_subplot(111)
# ax2.set_title("Aciertos por generación (Mutation rate: %s%%)" % (MUTATION_RATE*100))
# ax2.set_xlabel("# Generación")
# ax2.set_ylabel("% Aciertos")
# ax2.plot(generations1,success1, color="blue", label="Palabra de largo 3")
# ax2.plot(generations2,success2, color="red", label="Palabra de largo 26")
# ax2.plot(generations3,success3, color="y", label= "Palabra de largo 118")
# ax2.plot(generations4,success4, color='g', label= "Palabra de largo 309")
# plt.legend()
# plt.grid()
# plt.yticks(range(0, 101, 20))
# f2.show()

# plt.show()
# plt.close()