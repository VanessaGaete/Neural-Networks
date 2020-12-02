import matplotlib.pyplot as plt

from GeneticAlgorithm import report

def main(SOLUTION, GENES_CHOICES, GENES_NUMBER, SCORE_FUNCTION, POPULATION_NUMBER, MUTATION_RATE, NUMBER_OF_GENERATIONS, OPTIONS={}):
    ##GRAFICOS VARIANDO EL MUTATION RATE

    generations1, total_score1, success1=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,POPULATION_NUMBER,0.01,NUMBER_OF_GENERATIONS,OPTIONS)
    generations2, total_score2, success2=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,POPULATION_NUMBER,0.25,NUMBER_OF_GENERATIONS,OPTIONS)
    generations3, total_score3, success3=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,POPULATION_NUMBER,0.5,NUMBER_OF_GENERATIONS,OPTIONS)

    f1 = plt.figure(1)
    ax1 = f1.add_subplot(111)
    ax1.set_title("Score total poblacional (# Individuos: %s)" % POPULATION_NUMBER)
    ax1.set_xlabel("# Generación")
    ax1.set_ylabel("% Score total poblacional")
    ax1.plot(generations1,total_score1, color="blue", label="Mutation rate de 0.01")
    ax1.plot(generations2,total_score2, color="red", label="Mutation rate de 0.25")
    ax1.plot(generations3,total_score3, color='g', label="Mutation rate de 0.5")
    plt.legend()
    plt.grid()
    plt.yticks(range(0, 101, 20))
    f1.show()

    f2 = plt.figure(2)
    ax2 = f2.add_subplot(111)
    ax2.set_title("Aciertos por generación (# Individuos: %s)" % POPULATION_NUMBER)
    ax2.set_xlabel("# Generación")
    ax2.set_ylabel("% Aciertos")
    ax2.plot(generations1,success1, color="blue", label="Mutation rate de 0.01")
    ax2.plot(generations2,success2, color="red", label="Mutation rate de 0.25")
    ax2.plot(generations3,success3, color='g', label="Mutation rate de 0.5")
    plt.legend()
    plt.grid()
    plt.yticks(range(0, 101, 20))
    f2.show()

    ##GRAFICOS VARIANDO LA POBLACION

    generations1, total_score1, success1=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,20,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)
    generations2, total_score2, success2=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,50,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)
    generations3, total_score3, success3=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,75,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)
    generations4, total_score4, success4=report(SOLUTION,GENES_CHOICES,GENES_NUMBER,SCORE_FUNCTION,100,MUTATION_RATE,NUMBER_OF_GENERATIONS,OPTIONS)

    f1 = plt.figure(3)
    ax1 = f1.add_subplot(111)
    ax1.set_title("Score total poblacional (Mutation rate: %s%%)" % (MUTATION_RATE*100))
    ax1.set_xlabel("# Generación")
    ax1.set_ylabel("% Score total poblacional")
    ax1.plot(generations1,total_score1, color="blue", label="20 individuos")
    ax1.plot(generations2,total_score2, color="red", label="50 individuos")
    ax1.plot(generations3,total_score3, color='g', label="75 individuos")
    ax1.plot(generations4,total_score4, color='y', label="100 individuos")
    plt.legend()
    plt.grid()
    plt.yticks(range(0, 101, 20))
    f1.show()

    f2 = plt.figure(4)
    ax2 = f2.add_subplot(111)
    ax2.set_title("Aciertos por generación (Mutation rate: %s%%)" % (MUTATION_RATE*100))
    ax2.set_xlabel("# Generación")
    ax2.set_ylabel("% Aciertos")
    ax2.plot(generations1,success1, color="blue", label="20 individuos")
    ax2.plot(generations2,success2, color="red", label="50 individuos")
    ax2.plot(generations3,success3, color='g', label="75 individuos")
    ax2.plot(generations4,success4, color='y', label="100 individuos")
    plt.legend()
    plt.grid()
    plt.yticks(range(0, 101, 20))
    f2.show()

    plt.show()
    plt.close()