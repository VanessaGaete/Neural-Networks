import random

from Nodes import *

class GeneticProgram:
    def __init__(self, target: int, number_set: list, node_set: list, population_number: int, individual_depth: int, number_of_generations: int, mutation_rate: float, fitness_function):
        self.target = target
        self.number_set = number_set
        self.node_set = node_set
        self.population_number = population_number
        self.depth = individual_depth
        self.number_of_generations = number_of_generations
        self.mutation_rate = mutation_rate
        self.fitness_function = fitness_function
        
        self.population = []
        self.population_fitness = []

        self.history = []
        self.generationNumber = 0
        self.sol_gen = -1
        self.a_solution = None

        # First steps
        self.generate_population()
        self.calculate_population_fitness()
        self.checkSolutionExists()
    
    def Random_Generator(self, number_set, depth) -> AbstractNode:
        #Generates an individual with maximum depth "depth" where each of its nodes is randomly chosen from 
        # the set of possible nodes "self.node_set" and the numbers from "number_set"
        if number_set:
            n1 = random.choice(number_set)
        else:
            n1 = random.uniform(-5,5)
        
        if depth < 0:
            return Number(n1)
        else:
            Node_ = random.choice(self.node_set)
            
            if Node_ == Number:
                return Node_(n1)
            elif Node_ == X:
                return Node_()
            else:
                return Node_(self.Random_Generator(number_set, depth-1), self.Random_Generator(number_set, depth-1))
    
    def checkSolutionExists(self) -> None:
        # Checks if some individual is the solution of the problem, compairing its score with the success_value.
        if (self.sol_gen == -1 and 1 in self.population_fitness):
            self.sol_gen = self.generationNumber
            self.a_solution = self.population[self.population_fitness.index(1)]

    def generate_population(self) -> None:
        #Creates a population of population_number randomly generated individuals
        for _ in range(self.population_number):
            individual = self.Random_Generator(self.number_set, self.depth)
            self.population.append(individual)

    def calculate_population_fitness(self) -> None:
        #Calculate fitness for each individual in the population
        self.population_fitness = []
        for i in range(self.population_number):
            individual = self.population[i]
            self.population_fitness.append(self.fitness_function(individual, self.target))
            
    def crossover(self, father1: AbstractNode, father2: AbstractNode) -> AbstractNode:
        #Make a crossover between "father1" and "father2". 
        #For this, a random node in "father1" is chosen and replaced by a random node in "father2"
        new_tree=father1.copy()
        father1_list=father1.nodesList()
        father2_list=father2.nodesList()

        cross_point1 = random.randint(0,len(father1_list)-1)
        cross_point2 = random.randint(0,len(father2_list)-1)

        subtree=father2_list[cross_point2].copy()
        if cross_point1 == 0:
            return subtree

        new_tree.replace(cross_point1, subtree)

        return new_tree

    def mutation(self, individual) -> AbstractNode:
        #The individual mutates with some probability. 
        #If the individual mutates, a random node is replaced by a new randomly generated node.
        p = random.random()
        
        if (p < self.mutation_rate):

            i_list=individual.nodesList()
            mut_point = random.randint(0,len(i_list)-1)
            required_depth=self.depth-i_list[mut_point].depth
            random_subtree = self.Random_Generator(self.number_set, required_depth)

            if mut_point==0:
                return random_subtree
            
            individual.replace(mut_point, random_subtree)
        return individual
        
    def selectIndividual(self) -> AbstractNode:
        # Roulette Wheel selection
        s = sum(self.population_fitness)
        r = random.uniform(0,s)

        s=0
        for f, individual in zip(self.population_fitness, self.population):
            s+=f
            if (s>=r):
                return individual

    def step(self) -> None:
        # Steps one generation.
        # Uses tournament selection.
        new_generation = []
        for _ in range(self.population_number):
            new_individual = self.mutation(self.crossover(self.selectIndividual(), self.selectIndividual()))
            new_generation.append(new_individual)
        
        # Sets the new generation, calculates their fitness and adds 1 to generatino number.
        self.population = new_generation
        self.calculate_population_fitness()
        self.generationNumber += 1
        
        # Checks if this new generation was the first one that found the solution.
        self.checkSolutionExists()
            
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
                sum(self.population_fitness) / self.population_number * 100,
                self.totalSuccess()
            )
        )
    
    def totalSuccess(self) -> int:
        # Returns the current total succeses as percentage (Values between 0 and 1).
        return self.population_fitness.count(1) / self.population_number * 100

    def showSolution(self):
        # Prints the first ocurrence of the individual that is considered as a solution.
        try:
            print(self.a_solution)
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

def report(TARGET, NUMBER_SET, NODE_SET, INDIVIDUAL_DEPTH,FITNESS_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS):
    # Prints in console a report of the genetic algorithm after the last generation.
    # Also returns the GA statistics.

    gp = GeneticProgram(target=TARGET,
                    number_set=NUMBER_SET, 
                    node_set= NODE_SET, 
                    population_number= POPULATION_NUMBER, 
                    individual_depth= INDIVIDUAL_DEPTH, 
                    number_of_generations=NUMBER_OF_GENERATIONS,
                    mutation_rate=MUTATION_RATE, 
                    fitness_function=FITNESS_FUNCTION)

    for _ in range(NUMBER_OF_GENERATIONS):
        gp.step()
    
    print("\n--- REPORT ---")
    print("Input as solution:", TARGET)
    print("Individuals per generation:", POPULATION_NUMBER)
    print("Generations passed:", NUMBER_OF_GENERATIONS)
    print("Mutation rate:", MUTATION_RATE)
    print("Solution found after:", gp.sol_gen, "generations")
    print("Any valid solution:")
    gp.showSolution()
    print("--------------")
    
    return gp.stadistics()
