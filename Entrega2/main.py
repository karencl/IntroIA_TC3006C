import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def plotPCAs(pca_results, labels, final_title):
    """ Función para graficar los PCAs que representan todo el dataframe original
    Args:
        pca_results (DataFrame) - Dataframe con los 2 PCAs creados para la
                                  representación de los datos.
        labels (series) - serie que contiene los valores de los grupos reales
        final_title (string) - string que contiene una extensión para la última
                               parte del título de la gráfica  
    """
    
    plt.figure(figsize=(10,5))
    sns.scatterplot(x="pca1", y="pca2", hue=labels, data=pca_results)
    plt.title('K-means Clustering with 2 dimensions' + final_title)
    

def evaluateNumClusters(inputs):
    """ Función que utiliza 'elbow method' para evaluar el número apropiado de
        clusters que se deben de tener para este dataset
    Args:
        inputs (DataFrame) - Dataframe con todas las entradas
    """
    
    kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
    }

    sse = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(inputs)
        sse.append(kmeans.inertia_)
    
    plt.figure(figsize=(10,5))   
    plt.style.use("fivethirtyeight")
    plt.plot(range(1, 11), sse)
    plt.xticks(range(1, 11))
    plt.xlabel("Number of Clusters")
    plt.ylabel("SSE")
    plt.title("Elbow method")


if __name__ == '__main__':
    # Carga y manejo de datos
    seeds = pd.read_csv('seeds_dataset.csv')
    seeds

    inputs = seeds[['Area', 'Perimeter', 'Compactness', 
                    'Length of kernel', 'Width of kernel', 
                    'Asymmetry coefficient', 'Length of kernel grove']]
    labels = seeds[['Class']]

    # Creación de 2 PCAs para la representación del dataframe
    reduced_data = PCA(n_components = 2).fit_transform(inputs)
    results = pd.DataFrame(reduced_data,columns=['pca1','pca2'])

    # Se muestran nuevamente los datos con los agrupamientos reales
    plotPCAs(results, labels['Class'], " - GRUPOS REALES")

    # Inicialización de KMeans (3 clusters)
    kmeans = KMeans(
        n_clusters = 3,
        init="random",
        n_init=10,
        max_iter=300,
        random_state=42
    )
    
    # Fit para los inputs
    kmeans.fit(inputs)
    
    # E valor SSE más bajo
    print(kmeans.inertia_)
    # Posiciones finales de los centros
    print(kmeans.cluster_centers_)
    
    # Predicciones de agrupamiento
    y = kmeans.predict(inputs)
    
    # Se muestran nuevamente los datos, pero ahora con las predicciones
    plotPCAs(results, y, " - PREDICCIONES")
    
    # elbow method
    evaluateNumClusters(inputs)
    
    plt.show()