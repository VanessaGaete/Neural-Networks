"""
Esta es una función auxiliar que fue usada para obtener los valores máximos
y mínimos de la columna seleccionada
"""

import csv

maxV = 0
minV = 1000000000 # sabemos por inspección que no existe un valor mayor a este número

COLUMNA = 0

with open("6-star.csv") as csv_file:
  csv_reader = csv.reader(csv_file, delimiter=',')
  firstLine = True
  for row in csv_reader:
    if firstLine:  ## primera fila
      firstLine = False
    else:                ## resto de filas
      maxV = max(maxV, float(row[COLUMNA]))
      minV = min(minV, float(row[COLUMNA]))

print("Máximo", maxV)
print("Mínimo", minV)