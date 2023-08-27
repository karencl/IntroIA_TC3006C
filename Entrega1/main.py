# -*- coding: utf-8 -*-
"""
Momento de Retroalimentación: Implementación de una técnica de aprendizaje máquina sin el uso de un framework
M2 - Machine Learning

Karen Cebreros López
A01704254
"""

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def normalize(x, cols):
  """ Normalización de los valores de las columnas dentro de x
  Args:
    x (ndarray) - un arreglo que contiene todos los valores por columnas de las entradas
    cols (lst) - una lista que contiene todos los nombres de las columnas de las entradas
  
  Returns:
     x (ndarray) - un arreglo que contiene todos los valores de X normalizados
  """

  for col in cols:
    x[col] = (x[col] - x[col].min()) / (x[col].max() - x[col].min())

  return x


def gradientDescent(x, y, epochs, lr):
  """ Función del gradient descent
  Args:
    x (ndarray) - un arreglo que contiene los valores por columnas de las entradas
    y (ndarray) - un arreglo que contiene los valores de la salida
    epochs (int) - número de
    lr (int) - learning rate

  Returns:
    params (ndarray) - un arreglo que contiene los parámetros finales del modelo
    comparison_df (dataframe) - df que contiene las predicciones de Y, las salidas actuales
                                y los costos, de los datos de entrenamiento y los de prueba
    costs (list) - lista de costos
  """

  params = np.zeros((x.shape[1],1))
  costs = []
  n = len(x)

  for epoch in range(epochs):
    # Obtengo la hipótesis y el error
    hyp = np.dot(params.T, x.T)
    error = hyp - y.reshape(-1)

    # Obtengo el costo (MSE)
    cost = (1/2*n) * np.dot(error, error.T)
    costs.append(int(cost))

    # Obtengo gradiente (en la forma derivada del squared mean error)
    grad = lr * np.dot(error, x) * (1/n)

    # Actualizo parámetros
    params = (params.T - grad).T


  comparison_df = pd.DataFrame({'Y Hypothesis': np.squeeze(hyp.T),
                                'Y Real Values': np.squeeze(y),
                                'Errors': np.squeeze(error.T)})

  return params, comparison_df, costs


def fit(model, x, y):
  """ Función para hacer las predicciones
  Args:
    model (ndarray) - un arreglo que contiene los parámetros del modelo
    x (ndarray) - un arreglo que contiene los valores por columnas de las entradas
    y (ndarray) - un arreglo que contiene los valores de la salida

  Returns:
    fit_df (dataframe) - df que contiene las predicciones de Y, las salidas actuales
                         y los costos, de los datos de entrenamiento y los de prueba
  """

  n = len(x)

  y_predicted = np.dot(model.T, x.T)
  error = y_predicted - y.reshape(-1)
  cost = (1/2*n) * np.dot(error, error.T)

  fit_df = pd.DataFrame({'Fit - Y Predicted': np.squeeze(y_predicted.T),
                         'Fit - Y Real Values': np.squeeze(y),
                         'Fit - Errors': np.squeeze(error.T)})

  return fit_df


def toPlot(x, y1, y2, legend, title):
  """ Función para graficar resultados
  Args:
    x (ndarray) - un arreglo que contiene los valores de una entrada
    y1 (series) - serie que contiene las predicciones de la salida
    y2 (series) - serie que contiene los valores reales de la salida
    legend (list) - lista de las leyendas para la gráfica
    title (string) - título de la gráfica
  """
  
  plt.figure(figsize=(20,6))
  plt.scatter(x, y1, c ="pink", s = 10)
  plt.scatter(x, y2, c ="purple", s = 10)
  plt.legend(legend)
  plt.title(title)



if __name__ == '__main__':
  # Carga y manejo de datos
  cols = ['Temperature', 'Ambient Pressure', 'Relative Humidity', 'Exhaust Vacuum', 'Net hourly electrical energy']
  df_ccpp = pd.read_csv('combined_cycle_pp.csv')
  df_ccpp.columns = cols
  df_ccpp.head(10)

  df_ccpp = df_ccpp.sample(frac = 1, random_state=1)
  df_ccpp

  x = df_ccpp[cols[:-1]]
  x = normalize(x, cols[:-1])
  x = x.to_numpy()

  y = df_ccpp[cols[-1]].to_numpy()

  x_train = x[:8000]
  x_test = x[8000:]

  x_train = np.c_[np.ones(x_train.shape[0]), x_train]
  x_test = np.c_[np.ones(x_test.shape[0]), x_test]

  y_train = y[:8000]
  y_test = y[8000:]

  # Gradient descent algorithm using linear regression (model training)
  epochs = 2200
  lr = 0.03
  model, train_df, train_costs = gradientDescent(x_train, y_train, epochs, lr)
  _, _, test_costs = gradientDescent(x_test, y_test, epochs, lr)

  print("Final params: " + str(model.T))
  train_df

  # Predicciones
  fit_df = fit(model, x_test, y_test)
  fit_df

  # Comparaciones del costo entre entrenamiento y prueba
  plt.figure(figsize=(10,5))
  plt.plot(train_costs)
  plt.plot(test_costs)
  plt.legend(["train costs", "test costs"])
  plt.title("Costo del entrenamiento vs costo de prueba")


  # Visualización de gráficas del entrenamiento
  toPlot(x_train[:, [3]], train_df['Y Hypothesis'], 
         train_df['Y Real Values'], ["Hypothesis", "Y"], 
         (cols[2] + " vs hypotheis and y_train"))
  
  toPlot(x_train[:, [4]], train_df['Y Hypothesis'], 
         train_df['Y Real Values'], ["Hypothesis", "Y"], 
         (cols[3] + " vs hypotheis and y_train"))


  # Visualización de gráficas de las predicciones
  toPlot(x_test[:, [3]], fit_df['Fit - Y Predicted'], 
         fit_df['Fit - Y Real Values'], ["y_predicted", "Y"], 
         (cols[2] + " vs y_predicted and y_test"))
  
  toPlot(x_test[:, [4]], fit_df['Fit - Y Predicted'], 
         fit_df['Fit - Y Real Values'], ["y_predicted", "Y"], 
         (cols[3] + " vs y_predicted and y_test"))
  
  plt.show()
  