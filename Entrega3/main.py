# -*- coding: utf-8 -*-
"""
Momento de Retroalimentación: Análisis y Reporte sobre el desempeño del modelo
M2 - Machine Learning

Karen Cebreros López
A01704254
"""

# Librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


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
    hyp = params.T @ x.T
    error = hyp - y.reshape(-1)

    # Obtengo el costo (MSE)
    cost = (1/2*n) * (error @ error.T)
    costs.append(int(cost))

    # Obtengo gradiente (en la forma derivada del squared mean error)
    grad = lr * (error @ x) * (1/n)

    # Actualizo parámetros
    params = (params.T - grad).T


  comparison_df = pd.DataFrame({'Y Hypothesis': np.squeeze(hyp.T),
                                'Y Real Values': np.squeeze(y),
                                'Errors': np.squeeze(error.T)})

  return params, comparison_df, costs

def fitMyModel(model, x, y):
  """ Función para hacer las predicciones
  Args:
    model (ndarray) - un arreglo que contiene los parámetros del modelo
    x (ndarray) - un arreglo que contiene los valores por columnas de las entradas
    y (ndarray) - un arreglo que contiene los valores de la salida

  Returns:
    fit_df (dataframe) - df que contiene las predicciones de Y, las salidas actuales
                         y los costos, de los datos de entrenamiento y los de prueba
    cost (double) - costo
  """

  n = len(x)

  y_predicted = model.T @ x.T
  error = y_predicted - y.reshape(-1)
  cost = (1/n) * (error @ error.T)

  fit_df = pd.DataFrame({'Fit - Y Predicted': np.squeeze(y_predicted.T),
                         'Fit - Y Real Values': np.squeeze(y),
                         'Fit - Errors': np.squeeze(error.T)})

  return np.double(cost), fit_df


def toPlot(x, y1, y2, legend, title):
  """ Función para graficar resultados
  Args:
    x (ndarray) - un arreglo que contiene los valores de una entrada
    y1 (series) - serie que contiene las predicciones de la salida
    y2 (series) - serie que contiene los valores reales de la salida
    legend (list) - lista de las leyendas para la gráfica
    title (string) - título de la gráfica
  """

  plt.figure(figsize=(18,5))
  plt.scatter(x, y1, c ="pink", s = 10)
  plt.scatter(x, y2, c ="purple", s = 10)
  plt.legend(legend)
  plt.title(title)
  

if __name__ == '__main__':
  # Procesamiento de datos
  cols = ['Temperature', 'Ambient Pressure', 'Relative Humidity', 'Exhaust Vacuum', 'Net hourly electrical energy']
  df_ccpp = pd.read_csv('combined_cycle_pp.csv')
  df_ccpp.columns = cols
  df_ccpp.head(10)

  df_ccpp = df_ccpp.sample(frac = 1, random_state=1)
  df_ccpp

  # Normalización y separación de datos
  x = df_ccpp[cols[:-1]].copy()
  x = normalize(x, cols[:-1])
  x = x.to_numpy()

  y = df_ccpp[cols[-1]]
  y = y.to_numpy()

  x_train = x[:8000]
  x_test = x[8000:]

  x_train = np.c_[np.ones(x_train.shape[0]), x_train]
  x_test = np.c_[np.ones(x_test.shape[0]), x_test]

  y_train = y[:8000]
  y_test = y[8000:]

  # Creación y entrenamiento del modelo sin framework
  epochs = 2200
  lr = 0.03

  model_nf, train_df, train_costs = gradientDescent(x_train, y_train, epochs, lr)
  _, _, test_costs = gradientDescent(x_test, y_test, epochs, lr)

  # Creación y entrenamiento del modelo con framework
  model_f = LinearRegression(fit_intercept=True)
  model_f

  model_f.fit(x_train, y_train)

  # Predicciones sin framework
  final_cost_nf, fit_df = fitMyModel(model_nf, x_test, y_test)

  # Predicciones con framework
  pred_y = model_f.predict(x_test)
  pred_y
  
  # Comparación entre modelos
  # Parámetros de ambos modelos
  print(f'Final params - model without framework: {model_nf.T}')
  print(f'Final params - model with framework: {model_f.coef_}')

  # modelo con framework - r2
  aux = model_f.predict(x_train)
  print(f'r2 - train with framework: {r2_score(y_train, aux)}')
  print(f'r2 - test with framework: {r2_score(y_test, pred_y)}')
  
  # modelo sin framework - costo
  plt.figure(figsize=(10,5))
  plt.plot(train_costs)
  plt.plot(test_costs)
  plt.legend(["train costs", "test costs"])
  plt.title("Costo del entrenamiento vs costo de prueba del modelo sin framework")
  
  # Gráficas
  toPlot(x_test[:, [4]], pred_y, y_test, ["Predictions", "Real values"], 
        (cols[3] + " - y_predicted vs y_test - WITH FRAMEWORK"))
  
  toPlot(x_test[:, [4]], fit_df['Fit - Y Predicted'], 
        fit_df['Fit - Y Real Values'], ["Predictions", "Real values"], 
        (cols[3] + " - y_predicted vs y_test - WITHOUT FRAMEWORK"))
  
  toPlot(x_test[:, [3]], pred_y, y_test, ["Predictions", "Real values"], 
        (cols[2] + " - y_predicted vs y_test - WITH FRAMEWORK"))
  
  toPlot(x_test[:, [3]], fit_df['Fit - Y Predicted'], 
        fit_df['Fit - Y Real Values'], ["Predictions", "Real values"], 
        (cols[2] + " - y_predicted vs y_test - WITHOUT FRAMEWORK"))

  plt.show()