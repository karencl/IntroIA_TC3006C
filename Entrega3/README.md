# Análisis y reporte sobre el desempeño del modelo

## Descripción
Para el tercer entregable del módulo 2, decidí hacer el análisis sobre el dataset que utilicé para mi primer entregable, pero esta vez no solo lo implementé con el algoritmo de regresión lineal con gradiente descendiente, sino que también utilicé LinearRegression de la librería sklearn. Esto con el objetivo de poder hacer una comparativa de sus desempeños en cuanto a las predicciones. 

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

Para el entrenamiento dejé 8000 datos y para las predicciones 1568.

A ambos modelos les paso los mismos set de datos (los de *train* para entrenar el modelo y luego el de *test* para las predicciones), con el fin de poder comparar más adelante sus desempeños.

## Modelo sin framework
Este modelo es el modelo creado por mi para mi entrega 1 del módulo 2.
Para éste desarrollé funciones para normalizar, realizar el algoritmo de gradiente descendiente, hacer predicciones y hacer gráficas de resultados.

Los parámetros iniciales que pasé para entrenar al modelo fueron:
- 2000 épocas
- 0.03 de learning rate
- ndarray de x_train con 8000 datos (4 entradas)
- ndarray de y_train con 8000 datos (1 salida)

## Modelo con framework
Este modelo fue creado con la librería de *sklearn*. De esta importé *LinearRegression* para la creación del mismo modelo y *mean_squared_error* para poder visualizar el mse y compararlo con el del modelo sin framework.

En este caso, como tal no se pueden modificar los hiperparámetros, ya que la librería de *sklearn* no lo permite para este modelo en específico. Sin embargo, hice varias gráficas y saqué datos como el mse, para poder analizar su desempeño y compararlo con el otro modelo.

## Desempeño de ambos modelos
Primero que nada, comparemos algunos de las gráficas obtenidas de acuerdo a los valores reales de Y (de acuerdo al subset de entrenamiento) y las predicciones de Y (de acuerdo al subset de prueba), dependiendo de alguna de las entredas.

### Figuras 1 y 2 - Exhaust vacuum
- **Exhaust vacuum - y_predicted vs y_test - WITH FRAMEWORK:**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_1.png)

- **Exhaust vacuum - y_predicted vs y_test - WITHOUT FRAMEWORK:**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_2.png)

### Figuras 3 y 4 - Relative humidity
- **Relative humidity - y_predicted vs y_test - WITH FRAMEWORK:** 
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_3.png)

- **Relative humidity - y_predicted vs y_test - WITHOUT FRAMEWORK:**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_4.png)


Luego, pasemos a ver la diferencia que ambos modelos tienen, de acuerdo a su MSE:

### MSE
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Screen%20Shot%202023-09-07%20at%2016.49.53.png)
