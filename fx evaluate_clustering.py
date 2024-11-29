# Importaciones necesarias
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, cophenet
from sklearn.metrics.pairwise import euclidean_distances

# Función principal
def evaluate_clustering(clustering_type, model, data, metrics=None, model_name=None):
    """
    Evalúa la calidad de un modelo de clustering mediante diversas métricas.

    Args:
        clustering_type (str): Tipo de clustering ('k' para K-means, 'g' para clustering jerárquico).
        model: Modelo de clustering entrenado (ej., KMeans, AgglomerativeClustering).
        data (array-like): Datos utilizados para ajustar el modelo.
        metrics (pd.DataFrame, opcional): DataFrame existente para agregar las métricas calculadas.
        model_name (str, opcional): Nombre del modelo (ej., 'KMeans-5', 'Agglomerative-10').

    Returns:
        pd.DataFrame: DataFrame con las métricas calculadas para el modelo.
    """
    # Inicializar métricas con valores NaN
    metrics_dict = {
        'Model': [model_name],
        'Silhouette Coefficient': [np.nan],  #      Coeficiente de Silueta
        'Davies-Bouldin Index': [np.nan],    #      Índice de Davies-Bouldin
        'Calinski-Harabasz Index': [np.nan], #      Índice de Calinski-Harabasz
        'Cophenetic Coefficient': [np.nan],  #      Coeficiente cophenético (jerárquico)
        'WCSS': [np.nan],                    #      Suma de cuadrados dentro del cluster
        'BCSS': [np.nan],                    #      Suma de cuadrados entre clusters
        'Anomalies': [np.nan]                #      Número de anomalías detectadas
    }

    try:
        if clustering_type == 'k':
            #      Clustering K-means
            #      -----------------
            #      Realiza clustering K-means y calcula métricas específicas.

            clustering_results = model.labels_  # Etiquetas asignadas por el modelo
            centroids = model.cluster_centers_  # Coordenadas de los centroides
            squared_distances = np.square(euclidean_distances(data, centroids))

            # Métricas de calidad del clustering
            metrics_dict['Silhouette Coefficient'] = [silhouette_score(data, clustering_results)]  #      Mide cohesión y separación
            metrics_dict['Davies-Bouldin Index'] = [davies_bouldin_score(data, clustering_results)] #      Evalúa compacidad y separación
            metrics_dict['Calinski-Harabasz Index'] = [calinski_harabasz_score(data, clustering_results)] #      Relación entre dispersión intra/intercluster

            # Métricas específicas de K-means
            metrics_dict['WCSS'] = [np.sum(squared_distances[np.arange(len(data)), clustering_results])] #      Error cuadrático dentro del cluster
            metrics_dict['BCSS'] = [np.sum(squared_distances) - metrics_dict['WCSS'][0]]                 #      Dispersión entre clusters

            # Detección de anomalías (opcional)
            outliers = IsolationForest().fit_predict(data)  #      Detección de outliers usando Isolation Forest
            metrics_dict['Anomalies'] = [np.sum(outliers == -1)]  #      Cuenta el número de anomalías

        elif clustering_type == 'g':
            #      Clustering Jerárquico
            #      ---------------------
            #      Realiza clustering jerárquico y calcula métricas específicas.

            linkage_matrix = linkage(data, method='complete', metric='euclidean')  # Matriz de linkage
            metrics_dict['Cophenetic Coefficient'], _ = cophenet(linkage_matrix, pdist(data))  #      Correlación cophenética

            # Métricas de calidad del clustering
            metrics_dict['Silhouette Coefficient'] = [silhouette_score(data, model.labels_)]
            metrics_dict['Davies-Bouldin Index'] = [davies_bouldin_score(data, model.labels_)]
            metrics_dict['Calinski-Harabasz Index'] = [calinski_harabasz_score(data, model.labels_)]

            # Detección de anomalías
            outliers = LocalOutlierFactor().fit_predict(data)  #      Outliers usando Local Outlier Factor
            metrics_dict['Anomalies'] = [np.sum(outliers == -1)]

    except AttributeError as e:
        print(f"Error: {e}")

    # Crear un DataFrame con las métricas
    new_metrics = pd.DataFrame(metrics_dict)

    # Agregar métricas nuevas a un DataFrame existente o crear uno nuevo
    if metrics is None:
        metrics = new_metrics
    else:
        metrics = pd.concat([metrics, new_metrics], ignore_index=True)

    return metrics

# Notas sobre el uso:
# -------------------
# 1. Tipos de clustering soportados:
#    - 'k': K-means.
#    - 'g': Clustering jerárquico.
#
# 2. Métricas y su interpretación:
#    - Silhouette Coefficient: Varía entre -1 y 1. Valores cercanos a 1 indican clusters bien definidos.
#    - Davies-Bouldin Index: Valores bajos indican clusters compactos y bien separados.
#    - Calinski-Harabasz Index: Valores altos indican buena separación intercluster y cohesión intracluster.
#    - Cophenetic Coefficient: Evalúa cómo el dendrograma jerárquico preserva las distancias originales.
#    - WCSS (Within-Cluster Sum of Squares): Mide la compacidad de los clusters (más bajo es mejor).
#    - BCSS (Between-Cluster Sum of Squares): Representa la separación entre clusters (más alto es mejor).
#    - Anomalies: Número de puntos detectados como outliers o anomalías.