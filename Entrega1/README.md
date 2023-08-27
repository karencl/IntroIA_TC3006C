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

- **Gráfica de comparación de los costos entre los datos de entrenamiento y los de prueba:** En esta gráfica se puede observar que no hay una gran diferencia entre ambos costos, por lo que podemos deducir que no hay overfitting en el modelo. Cabe aclarar que la línea naranja (los costos de los datos de prueba), sale más pequeña debido al tamaño de ese arreglo, a comparación del tamaño de los datos de entrenamiento.

- **Gráficas de comparación entre la hipótesis y valores reales de Y (estas son del entrenamiento):** En estas dos gráficas se puede observar que hay una gran cantidad de datos (8000 en total) y podemos ver como es que los valores reales de la salida, se relacionan con una de las columnas de entrada. En éstas se muestra como mi modelo busca crear una línea en diagonal (la regresión lineal), para abarcar la mayoría de los datos reales.

- **Gráficas de comparación entre las predicciones y valores reales de Y (estas son de las pruebas):** En estas otras dos gráficas se puede observar una cantidad menor de datos (1568 en total) y de igual forma podemos ver como es que los valores reales de la salida, se relacionan con una de las columnas de entrada. En éstas dos a comparación de las anteriores, se ve como es que las predicciones de Y (ya con los parámetros obtenidos del entrenamiento), van encajando mejor con los valores reales de la salida.