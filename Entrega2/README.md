## Técnica utilizado
Para la segunda entrega del módulo 2, decidí implementar KMeans para el agrupamiento de los datos del dataset que elegí.

## Dataset
El dataset que utilicé para este entregable se llama "Seeds", obtenido de: https://archive.ics.uci.edu/dataset/236/seeds.
Lo que se busca con este data set es agrupar tres tipos diferentes de trigo: Kama, Rosa and Canadian. Esto se hace mediante los datos que se obtienen (área, perímetro, compacidad, largo, ancho, etc), al analizar el núcleo de éstos, mediante una técnica de rayos x.

Especificaciones:
- 210 datos
- Multivariable
- No. de atributos: 7
- Tipo de artibutos: Reales
- No. de grupos: 3

## Especificaciones
Para la visualización de los datos, apliqué la técnica de PCAs, para poder visualizar todas las entradas del dataset de manera representativa en una sola gráfica.
***(NOTA: esta técnica la utilicé meramente para visualización y no para entrenar al modelo)***


## Observaciones de los resultados

- **Gráfica 1 de los PCAs (grupos reales):**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega2/Images/Figure_1.png)

En esta gráfica se puede observar el dataset completo de manera representativa, utilizando la técnica de PCAs. En esta podemos ver cuales son los 3 grupos reales y como los datos se distribuyen entre ellos.

- **Gráfica 2 de los PCAs (predicciones):**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega2/Images/Figure_2.png)

En esta gráfica también se puede observar el dataset completo de manera representativa, utilizando la técnica de PCA. Sin embargo, la diferencia que tiene ésta con la gráfica pasada, es que podemos ver cuales son los 3 grupos que se predijeron y como los datos que se asignaron a cada uno de ellos se distribuyen.

Podemos observar que los grupos que se predijeron (junto con los datos que hay en cada uno de ellos), son bastante similares a los grupos reales (y sus datos) mostrados en la gráfica 1 de los PCAs; por lo que se puede decir que el uso del método 'KMeans' cumplió con el objetivo que se buscaba en esta práctica.

- **Gráfica del 'elbow method':**
![alt text](https://github.com/karencl/IntroIA_TC3006C/blob/master/Entrega2/Images/Figure_3.png)

En esta gráfica se pueden observar el resultado obtenido al utilizar la técnica 'elbow method', que sirve para evaluar el número adecuado de clusters que deberían de haber para este dataset y como podemos ver, el 'elbow' está prácticamente en 3.
