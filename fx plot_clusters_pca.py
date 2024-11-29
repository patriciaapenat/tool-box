# Importaciones necesarias
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función principal
def plot_cluster_pca(data, labels, cluster_centers, title=None, xlabel=None, ylabel=None):
    """
    Visualiza los clusters en un espacio reducido mediante PCA, en 2D y 3D.

    Args:
        data (array-like): Datos originales (dimensionalidad completa).
        labels (array-like): Etiquetas de cluster para cada punto de datos.
        cluster_centers (array-like): Coordenadas de los centros de los clusters.
        title (str, opcional): Título para el gráfico.
        xlabel (str, opcional): Etiqueta para el eje X en el gráfico 2D.
        ylabel (str, opcional): Etiqueta para el eje Y en el gráfico 2D.

    Returns:
        None. La función genera gráficos 2D y 3D directamente.
    """
    # Reducción de dimensionalidad con PCA
    # ------------------------------------
    # PCA a 2 componentes para visualización 2D
    pca_2d = PCA(n_components=2)
    data_pca_2d = pca_2d.fit_transform(data)
    
    # PCA a 3 componentes para visualización 3D
    pca_3d = PCA(n_components=3)
    data_pca_3d = pca_3d.fit_transform(data)
    
    # Proyectar los centroides en el espacio PCA
    pca_centers_2d = pca_2d.transform(cluster_centers)
    pca_centers_3d = pca_3d.transform(cluster_centers)

    # Convertir los datos a DataFrame para manipulación más sencilla
    df_2d = pd.DataFrame(data_pca_2d, columns=['PC1', 'PC2'])
    df_2d['Cluster'] = labels.astype(str)
    
    df_3d = pd.DataFrame(data_pca_3d, columns=['PC1', 'PC2', 'PC3'])
    df_3d['Cluster'] = labels.astype(str)
    
    # Gráfico 2D
    plt.figure(figsize=(12, 6))  # Tamaño de la figura para 2D y 3D
    ax1 = plt.subplot(121)  # Subgráfico para 2D a la izquierda
    scatter = ax1.scatter(data_pca_2d[:, 0], data_pca_2d[:, 1], c=labels, cmap='viridis', s=10, alpha=0.5)
    ax1.scatter(pca_centers_2d[:, 0], pca_centers_2d[:, 1], marker='*', c='red', s=200, label='Centroids')
    if title:
        ax1.set_title(title + ' - 2D')
    if xlabel:
        ax1.set_xlabel(xlabel)
    if ylabel:
        ax1.set_ylabel(ylabel)
    legend1 = ax1.legend(*scatter.legend_elements(), title="Clusters")
    ax1.add_artist(legend1)

    # Gráfico 3D
    ax2 = plt.subplot(122, projection='3d')
    ax2.scatter(data_pca_3d[:, 0], data_pca_3d[:, 1], data_pca_3d[:, 2], c=labels, cmap='viridis', s=10, alpha=0.5)
    ax2.scatter(pca_centers_3d[:, 0], pca_centers_3d[:, 1], pca_centers_3d[:, 2], marker='*', c='red', s=200, label='Centroids')
    if title:
        ax2.set_title(title + ' - 3D')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    ax2.legend()

    plt.tight_layout()  # Ajustar los elementos del gráfico para que no se superpongan
    plt.show()

# Notas sobre el uso:
# -------------------
# 1. Argumentos de entrada:
#    - data: Matriz de datos original. Debe tener dimensiones (n_samples, n_features).
#    - labels: Etiquetas de cluster asignadas a cada punto por el modelo de clustering.
#    - cluster_centers: Coordenadas de los centroides de los clusters.
#    - title: (Opcional) Título para ambos gráficos.
#    - xlabel, ylabel: (Opcional) Etiquetas para los ejes del gráfico 2D.
#
# 2. ¿Qué es PCA?
#    - PCA (Análisis de Componentes Principales) reduce la dimensionalidad de los datos,
#      proyectándolos en un nuevo espacio donde las primeras componentes capturan la mayor varianza.
#    - En este caso, se usa PCA para reducir los datos a 2 y 3 dimensiones para visualización.
#
# 3. Explicación de los gráficos:
#    - Gráfico 2D:
#      - Proyecta los puntos de datos en un plano (PC1 vs PC2).
#      - Los centroides de los clusters aparecen como estrellas rojas ('*').
#    - Gráfico 3D:
#      - Agrega una dimensión adicional (PC3) para mostrar una vista tridimensional.
#      - Los clusters pueden observarse en el espacio tridimensional.
#
# 4. Recomendaciones:
#    - Úsalo cuando quieras visualizar clusters en datasets con más de 3 dimensiones.
#    - Requiere que el modelo de clustering proporcione etiquetas (`labels`) y centroides (`cluster_centers`).
