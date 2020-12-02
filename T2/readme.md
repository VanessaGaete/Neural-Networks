# README

Para cada problema se genera un reporte con los siguientes datos:

    • Input as solution: correspondiente a la palabra que se le pidió buscar
    • Individuals per generation: número de individuos en la población
    • Generations passed: la cantidad de generaciones que se le seseó al algoritmo
    • Mutation rate
    • Solution found after: el número de la generación donde se encontró una solución
    • Other options
    • Any valid solution: se muestra cualquier individuo que sea correcto.

Por otro lado, cada programa viene con la posibilidad de ver gráficos de comparación para los hiperparámetros. Esto cual genera 4 gráficos para el problema. Dos correspondientes al score poblacional y al porcentaje de aciertos de los mismos algoritmos pero con distintos mutation rate, y los otros dos correspondientes al score poblacional y al porcentaje de aciertos con distinto número de individuos por población.

Además en cada programa hay también una sección para generar gráficos de comparación para distintos ejemplos de solución. Esto genera dos gráficos, uno para el score poblacional y otro para el porcentaje de aciertos.

## Finding a Word

Para ejecutar este problema se debe correr el archivo EX1_FindAWord.py. Esto mostrará en la consola el reporte ya descrito. Inicialmente el programa viene seteado para buscar la palabra “lorem”, una población de 75 individuos, mutation rate de 0.25 y 100 generaciones. En caso de querer probar con otros parámetros, se deben modificar en ese mismo archivo las variables:

    • SOLUTION
    • POPULATION_NUMBER
    • MUTATION_RATE
    • NUMBER_OF_GENERATIONS

En caso de querer ver gráficos para distintos hiperparámetros se debe descomentar la línea 34 que llama a ReportUtils.

Por otro lado si se quieren ver los gráficos de comparación para distintas palabras se debe descontar toda la sección de “ESTUDIO DE DISTINTOS VALORES PARA EL PROBLEMA” que esta a partir de la linea 36.

## Conversión de decimal a binario

Para ejecutar este programa se debe correr el archivo EX2_DecimalToBinary.py. Esto mostrará en la consola el reporte ya descrito. Inicialmente el programa viene seteado para convertir el número 300, una población de 75 individuos, mutation rate de 0.25 y 100 generaciones. En caso de querer probar con otros parámetros, se deben modificar en ese mismo archivo las variables:

    • SOLUTION
    • POPULATION_NUMBER
    • MUTATION_RATE
    • NUMBER_OF_GENERATIONS

En caso de querer ver gráficos para distintos hiperparámetros se debe descomentar la línea 35 que llama a ReportUtils.

Por otro lado si se quieren ver los gráficos de comparación para distintas palabras se debe descontar toda la sección de “ESTUDIO DE DISTINTOS VALORES PARA EL PROBLEMA” que esta a partir de la linea 37.

## N Reinas

Para ejecutar este programa se debe correr el archivo N_Queens.py. Esto mostrará en la consola el reporte. Inicialmente el programa viene seteado para posicionar 4 reinas en un tablero de 4x4, una población de 75 individuos, mutation rate de 0.25 y 100 generaciones. En caso de querer probar con otros parámetros, se deben modificar en ese mismo archivo las variables:

    • SOLUTION
    • DIMENSION
    • POPULATION_NUMBER
    • MUTATION_RATE
    • NUMBER_OF_GENERATIONS

En caso de querer ver gráficos para distintos hiperparámetros se debe descomentar la línea 161 que llama a ReportUtils.

Por otro lado si se quieren ver los gráficos de comparación para distintas palabras se debe descontar toda la sección de “ESTUDIO DE DISTINTOS VALORES PARA EL PROBLEMA” que esta a partir de la linea 163.