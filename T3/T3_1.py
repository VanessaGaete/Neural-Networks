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






TARGET = 0.5
NODE_SET = [Add, Mult, Div, Subs, Number]
FITNESS_FUNCTION = individual_fitness_DCDL

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
    # ReportUtils.main(TARGET, NUMBER_SET, NODE_SET, INDIVIDUAL_DEPTH,FITNESS_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS)