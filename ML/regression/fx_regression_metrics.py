# Importaciones necesarias
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, max_error
import math

# Función para evaluar modelos de regresión
def metriques(titulo, df, ytest, prediccio):
    """
    Evalúa un modelo de regresión mediante diversas métricas.

    Args:
        titulo (str): Nombre o descripción del modelo evaluado.
        df (list): Lista existente para agregar las métricas calculadas.
        ytest (array-like): Valores reales de la variable objetivo.
        prediccio (array-like): Valores predichos por el modelo.

    Returns:
        list: Lista actualizada con las métricas calculadas.
    """
    # Calcular R² (Coeficiente de determinación)
    # Mide qué proporción de la varianza de la variable dependiente es explicada por el modelo.
    r2 = round(r2_score(ytest, prediccio), 2)

    # Calcular MSE (Error Cuadrático Medio)
    # Promedio de los errores al cuadrado. Penaliza errores grandes.
    mse = round(mean_squared_error(ytest, prediccio), 2)

    # Calcular RMSE (Raíz del Error Cuadrático Medio)
    # Raíz cuadrada del MSE. Representa el error promedio en las mismas unidades que la variable objetivo.
    rmse = round(math.sqrt(mean_squared_error(ytest, prediccio)), 2)

    # Calcular MAE (Error Absoluto Medio)
    # Promedio de los errores absolutos. Menos sensible a valores atípicos que el MSE.
    mae = round(mean_absolute_error(ytest, prediccio), 2)

    # Calcular MAPE (Error Absoluto Porcentual Medio)
    # Proporción promedio del error relativo respecto a los valores reales.
    mape = round((mean_absolute_error(ytest, prediccio) / max(1e-10, sum(ytest) / len(ytest))) * 100, 2)  # Evitar división por cero

    # Calcular Mediana del Error Absoluto
    # Mide la mediana de los errores absolutos. Robusto frente a valores atípicos.
    mediana_ae = round(median_absolute_error(ytest, prediccio), 2)

    # Calcular Máximo Error
    # Error más alto entre los valores predichos y reales.
    max_err = round(max_error(ytest, prediccio), 2)

    # Agregar las métricas calculadas a la lista
    df.append([
        titulo,                # Nombre del modelo
        r2,                   # R²
        "{} %".format(round(r2 * 100, 2)),  # R² como porcentaje
        mse,                  # MSE
        rmse,                 # RMSE
        mae,                  # MAE
        "{} %".format(mape),  # MAPE como porcentaje
        mediana_ae,           # Mediana del Error Absoluto
        max_err               # Máximo Error
    ])

    return df

# Notas sobre el uso:
# -------------------
# 1. Métricas calculadas:
#    - R²: Proporción de varianza explicada. Varía entre -∞ y 1 (más alto es mejor).
#    - MSE: Error promedio al cuadrado. Penaliza errores grandes (más bajo es mejor).
#    - RMSE: Raíz cuadrada del MSE. Interpretación directa en las unidades originales.
#    - MAE: Error promedio absoluto. Indica el error promedio sin penalizar valores atípicos.
#    - MAPE: Porcentaje promedio del error relativo. Útil para comparar con otras variables.
#    - Mediana del Error Absoluto: Robusto frente a valores atípicos.
#    - Máximo Error: Útil para identificar el peor caso en las predicciones.

# 2. Entrada esperada:
#    - 'df' debe ser una lista que contendrá los resultados. Alternativamente, puede transformarse a un DataFrame después de varias iteraciones.