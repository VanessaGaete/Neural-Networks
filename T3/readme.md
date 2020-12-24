# README

### Librería de nodos

Esta se encuentra en Nodes.py
Se hace uso de una clase abstracta llamada AbstractNode, creada para crear los métodos más generales de cada nodo como reemplazar un nodo por otro, o evaluar su contenido.

Todos los nodos específicos hacen override de copy() cuando se necesita pedir una copia del nodo, además, para el caso de X, se recibe un parámetro opcional que reemplaza su valor por el valor pasado al string, esto con el fin de evaluar la operación completa.

### Consideraciones generales por ejercicio

Para cada problema, ubicados en T3_1.py y T3_2.py respectivamente, se genera un reporte con los siguientes datos:

    • Input as solution: Un entero para el ejercicio 1, y una tupla de puntos para el 2.
    • Individuals per generation: número de individuos en la población.
    • Generations passed: la cantidad de generaciones que se le pide al algoritmo.
    • Mutation rate: Un flotante entre 0 y 1 con la probabilidad de mutar.
    • Solution found after: el número de la generación donde se encontró una solución.
    • Any valid solution: para esta tarea, muestra la primera solución encontrada.

Por otro lado, cada programa (siempre aludiendo a T3_1 y T3_2) viene con la posibilidad de ver gráficos de comparación para los hiperparámetros. Esto cual genera 4 gráficos para el problema. Dos correspondientes al score poblacional y al porcentaje de aciertos de los mismos algoritmos pero con distintos mutation rate, y los otros dos correspondientes al score poblacional y al porcentaje de aciertos con distinto número de individuos por población.

Además, en cada programa hay una sección para generar gráficos de comparación para distintos ejemplos de solución. Esto genera dos gráficos, uno para el score poblacional y otro para el porcentaje de aciertos.

En la línea 28 es posible cambiar "INDIVIDUAL_DEPTH" para generar nodos de mayor o menor profundidad, esto afectará tanto la profunidad al crear nodos de la población original, como al mutar.

Por otro lado si no se quieren ver los gráficos de comparación para distintos parámetros se debe comentar la línea 49 que genera el reporte. De esta manera solo se prueba el código con un target por ejecución.

Para ejecutar los problemas se debe correr el archivo T3_1.py o T3_2.py . Esto mostrará en la consola el reporte ya descrito previamente. Inicialmente el programa ya viene con una solución para testear, una población de 50 individuos, mutation rate de 0.25 y 50 generaciones. En caso de querer probar con otros parámetros, se deben modificar en ese mismo archivo las variables:

    • TARGET (líneas 19 a/o 21)
    • POPULATION_NUMBER (línea 20)
    • NUMBER_OF_GENERATIONS (28)
    • MUTATION_RATE (línea 29)

## Ejercicio 1: "Des chiffres et des lettres"

T3_1.py

Se usan todos los nodos existentes para encontrar una solución, a excepción del nodo X. Si se desea, se puede remover alguno de los nodos para encontrar soluciones alternas con una cantidad limitada de operaciones binarias.

El target default es 10 y este valor puede ser tanto entero como flotante.
La fitness function debe permanecer inalterable.


## Ejercicio 2: Encontrar función que pase por los puntos

T3_2.py

Se usan todos los nodos existentes para encontrar una solución. Si se desea, se puede remover alguno de los nodos para encontrar soluciones alternas con una cantidad limitada de operaciones binarias.

El target default es una tupla de puntos que simulan el lado derecho de una parábola. Existen en las siguientes líneas otros targets para probar. Los puntos pueden ser tanto flotantes como enteros, sin embargo, se debe notar que la función encontrada siempre busca pasar por los puntos, si no es posible, ninguna solución será encontrada

La fitness function debe permanecer inalterable.