## Algoritmo utilizado
Para la primera entrega del módulo 2, decidí implementar el algoritmo de Gradient Descent con regresión lineal.

## Dataset
El dataset que utilicé para este entregable se llama "Combined Cycle Power Plant", obtenido de: https://archive.ics.uci.edu/dataset/294/combined+cycle+power+plant.
Lo que se busca con este data set es encontrar la salida neta de energía eléctrica por hora, dependiendo de los datos recopilados en un lapso de 6 años, cuando la planta estaba operando a máxima capacidad. Dicha salida depende de 4 variables ambientales: la temperatura, la presión ambiental, la humedad relativa y el vacío de escape; las cuales van cambiando dependiendo al funcionamiento de las turbinas de gas y vapor que se encuentran en la planta.

Especificaciones:
- 9,568 datos
- Multivariable
- No. de atributos: 4
- Tipo de artibutos: Reales
- No. de salidas: 1

## Especificaciones de uso del dataset para el entrenaimiento
Hice un shuffle del dataset antes de normalizar los datos de entrada. Una vez hecho esto, dividí el dataset para tener mis listas de entrenamiento y prueba, en X y Y. 

Para el entrenamiento dejé 9000 datos y para las predicciones 568.

## Observaciones de los resultados
