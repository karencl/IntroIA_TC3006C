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
Este modelo fue creado con la librería de *sklearn*. De esta importé *LinearRegression* para la creación del mismo modelo y *r2_score* para poder visualizar el mse y compararlo con el del modelo sin framework.

En este caso, como tal no se pueden modificar los hiperparámetros, ya que la librería de *sklearn* no lo permite para este modelo en específico. Sin embargo, hice varias gráficas y saqué datos como el r2, para poder analizar su desempeño y compararlo con el otro modelo.

## Desempeño de ambos modelos
Primero que nada, comparemos algunos de las gráficas obtenidas de acuerdo a los valores reales de Y (de acuerdo al subset de entrenamiento) y las predicciones de Y (de acuerdo al subset de prueba), dependiendo de alguna de las entredas.

### Figuras 1 y 2 - Exhaust vacuum
- **Exhaust vacuum - y_predicted vs y_test - WITH FRAMEWORK:**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_1.png)

- **Exhaust vacuum - y_predicted vs y_test - WITHOUT FRAMEWORK:**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_2.png)

En estas dos gráficas podemos ver como es que se encuentran las predicciones y los valores reales, dependiendo de una de las variables de entrada: exhaust vacuum.
La primera gráfica muestra las predicciones obtenidas a través del modelo creado con un framework y en la segunda, las predicciones que obtuve con el modelo que yo hice. Como se puede observar, las predicciones de la primera gráfica son mucho más acertadas que las de la segunda gráfica; lo que nos da la primera pista de que el modelo que usa el framework, es mejor que el que no lo tiene.

### Figuras 3 y 4 - Relative humidity
- **Relative humidity - y_predicted vs y_test - WITH FRAMEWORK:** 
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_3.png)

- **Relative humidity - y_predicted vs y_test - WITHOUT FRAMEWORK:**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega3/Images/Figure_4.png)

Ahora como una segunda comprobación a la suposición que hice a partir de las dos gráficas anteriores (que el modelo con framework es mejor que el de sin framework), realicé ptras dos gráficas que de igual manera comparan las predicciones y los valores reales, pero ahora dependiendo de la variable de entrada: relative humidity.
Y tal y como podemos ver en ellas, nuevamente las predicciones hechas con el modelo con framework, se acercan mucho más a los valores reales, que aquellas obtenidas con el modelo sin framework.


Ahora, para continuar con el análisis de desempeño de ambos modelos, pasemos a ver la diferencia que ambos modelos tienen, de acuerdo con el r2 del modelo con framework y las gráficas de costo del modelo sin framework:

### r2 - modelo con framework
- ![alt text]()



### costos - modelo sin framework
- ![alt text]()
