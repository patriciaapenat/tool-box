# Importaciones necesarias
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

def describe_clusters(df_data, km):
    """
    Describe los clusters generados por un modelo de clustering.

    Args:
        df_data (pd.DataFrame): Datos de entrada con asignaciones de cluster.
        km: Modelo de clustering entrenado (ej., KMeans).

    Returns:
        list: Lista de descripciones de cada cluster.
    """
    # Identificar las columnas continuas y categóricas
    continuous_cols = ['continuous_feature_1', 'continuous_feature_2', 'continuous_feature_3']
    categorical_cols = ['categorical_feature_1', 'categorical_feature_2', 'categorical_feature_3']
    discount_col = 'discount_feature'  # Columna genérica para valores de descuento

    descriptions = []
    centroids = km.cluster_centers_
    
    # Calcular el total de una métrica general (por ejemplo, ingresos)
    total_metric = df_data['metric_column'].sum()
    
    for cluster in np.unique(df_data['cluster']):
        # Filtrar los datos del clúster actual
        cluster_data = df_data[df_data['cluster'] == cluster]
        n_records = len(cluster_data)
        description = f"Cluster {cluster} (n={n_records}): \n"
        
        # Calcular la suma de la métrica del cluster
        cluster_metric = cluster_data['metric_column'].sum()
        description += f" - Suma de la métrica del cluster: {cluster_metric:.2f}\n"
        
        # Calcular el peso del cluster
        cluster_weight = (cluster_metric / total_metric) * 100
        description += f" - Peso del cluster: {cluster_metric:.2f} ({cluster_weight:.2f}%)\n"
        
        # Calcular la métrica media por cliente del cluster
        cluster_metric_avg = cluster_metric / n_records
        description += f" - Métrica media por cliente del cluster: {cluster_metric_avg:.2f}\n"
        
        # Calcular el porcentaje que representa cada cliente respecto al total
        cluster_data['metric_percentage'] = (cluster_data['metric_column'] / total_metric) * 100
        metric_percentage_avg = cluster_data['metric_percentage'].mean()
        description += f" - % de la métrica que representa cada cliente del cluster respecto al total: {metric_percentage_avg:.2f}%\n"

        # Describir las características continuas
        continuous_description = ""
        for col in continuous_cols:
            if col in df_data.columns:
                try:
                    col_min = cluster_data[col].min()
                    col_max = cluster_data[col].max()
                    col_mean = cluster_data[col].mean()
                    col_median = cluster_data[col].median()
                    col_decile = cluster_data[col].quantile([0.1, 0.9])
                    
                    continuous_description += (
                        f" - {col}: (media: {col_mean:.2f}, mediana: {col_median:.2f}, "
                        f"min: {col_min:.2f}, max: {col_max:.2f}, decil 10-90: {col_decile[0.1]:.2f}-{col_decile[0.9]:.2f})\n"
                    )
                except Exception as e:
                    continuous_description += f" - {col}: Error al calcular estadísticas ({str(e)})\n"
        
        # Tratar la columna de descuento si existe
        if discount_col in df_data.columns:
            try:
                discount_mean = cluster_data[discount_col].mean()
                discount_min = cluster_data[discount_col].min()
                discount_max = cluster_data[discount_col].max()
                continuous_description += f" - {discount_col}: (media: {discount_mean:.2f}, min: {discount_min:.2f}, max: {discount_max:.2f})\n"
            except Exception as e:
                continuous_description += f" - {discount_col}: Error al calcular estadísticas ({str(e)})\n"
        description += continuous_description

        # Describir las características categóricas
        categorical_description = ""
        for col in categorical_cols:
            if col in df_data.columns:
                value_counts = cluster_data[col].value_counts(normalize=True)
                for value, freq in value_counts.head(3).items():
                    categorical_description += f" - {col}: valor {value} (frecuencia: {freq:.2%})\n"
        description += categorical_description

        # Añadir la descripción del cluster a la lista de descripciones
        descriptions.append(description)

    return descriptions

# Notas sobre el uso:
# 1. Estructura de datos esperada:
#    - El DataFrame `df_data` debe contener una columna llamada `cluster`, que indica el cluster asignado para cada fila.
#    - Debe incluir una columna de métrica continua para el análisis general (configurable como `metric_column`).
#    - Las columnas continuas y categóricas deben definirse en las listas `continuous_cols` y `categorical_cols`, respectivamente.

# 2. Adaptaciones requeridas:
#    - Cambia los nombres en `continuous_cols`, `categorical_cols` y `discount_col` para que coincidan con tu conjunto de datos.
#    - Asegúrate de que las columnas necesarias existan en el DataFrame `df_data`.

# 3. Modelo de clustering:
#    - La función está diseñada para trabajar con modelos como `KMeans` de `sklearn`.
#    - El modelo debe proporcionar los centros de los clusters a través de `km.cluster_centers_`.

# 4. Salida:
#    - Devuelve una lista de strings, donde cada string contiene una descripción detallada de un cluster.
#    - Las descripciones incluyen estadísticas continuas, categóricas y porcentajes sobre la métrica principal.
