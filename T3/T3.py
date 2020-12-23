import random

from GeneticProgram import GeneticProgram
import ReportUtils
from Nodes import *
        
def individual_fitness_DCDL(individual, target):
    #The fitness function for Des chiffres et des lettres problem.
    try:
        i_value = individual.eval()
    except ZeroDivisionError:
        return 0

    score = 1 / (abs(target-i_value) + 1)
    return score

def individual_fitness_functions(individual, points_list):
    #The fitness function for the problem of finding an equation that fits three points.
    score=0
    for point in points_list:
        x = point[0]
        y = point[1]
        try:
            i_value = individual.eval(x)
            score += 1 / (abs(y-i_value) + 1)
        except ZeroDivisionError:
            score += 0
    return score / len(points_list)

# Descomentar para el ejercicio 1

TARGET = 10                                   # Ejercicio 1
NODE_SET = [Add, Mult, Div, Subs, Number]     # Ejercicio 1
FITNESS_FUNCTION = individual_fitness_DCDL    # Ejercicio 1

# Descomentar para el ejercicio 2

# TARGET = ((1,1), (2,2), (3,3),)                 # Ejercicio 2 / X
# TARGET = ((1,2), (2,4), (3,6),)                 # Ejercicio 2 / 2*X
# TARGET = ((1,1), (2,4), (3,9),(4,16))           # Ejercicio 2 / X^2
# NODE_SET = [Add, Mult, Div, Subs, Number, X]    # Ejercicio 2
# FITNESS_FUNCTION = individual_fitness_functions # Ejercicio 2



NUMBER_SET = [0,1,2,3,4,5,6,7]

POPULATION_NUMBER = 20
INDIVIDUAL_DEPTH = 3

NUMBER_OF_GENERATIONS = 50
MUTATION_RATE = 0.25

if __name__ == "__main__":
    dcdl = GeneticProgram(TARGET, NUMBER_SET, NODE_SET, POPULATION_NUMBER, INDIVIDUAL_DEPTH, NUMBER_OF_GENERATIONS, MUTATION_RATE, FITNESS_FUNCTION)
    
    for _ in range(NUMBER_OF_GENERATIONS):
        dcdl.step()
    
    print("\n--- REPORT ---")
    print("Input as solution:", TARGET)
    print("Individuals per generation:", POPULATION_NUMBER)
    print("Generations passed:", NUMBER_OF_GENERATIONS)
    print("Mutation rate:", MUTATION_RATE)
    print("Solution found after:", dcdl.sol_gen, "generations")
    print("Any valid solution:")
    dcdl.showSolution()
    print("--------------")
    

    ###GENERA UN REPORTE DE COMPARACION PARA DISTINTOS HIPERPARAMETROS###########
    ReportUtils.main(TARGET, NUMBER_SET, NODE_SET, INDIVIDUAL_DEPTH,FITNESS_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS)