# Importaciones necesarias
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, roc_auc_score
)
from sklearn.model_selection import cross_val_score

# Función para calcular métricas de clasificación
def obtener_metricas(nombre, transformacion, descripcion, y_test, modelo, X_test, X_train, y_train, df=None):
    """
    Calcula métricas de clasificación y las almacena en un DataFrame.

    Args:
        nombre (str): Nombre del modelo o experimento.
        transformacion (str): Descripción de transformaciones aplicadas.
        descripcion (str): Información adicional sobre el modelo o datos.
        y_test (array-like): Valores reales de la variable objetivo para el conjunto de prueba.
        modelo: Modelo de clasificación ya entrenado.
        X_test (array-like): Datos de prueba.
        X_train (array-like): Datos de entrenamiento.
        y_train (array-like): Etiquetas de entrenamiento.
        df (pd.DataFrame, opcional): DataFrame existente para almacenar las métricas calculadas.

    Returns:
        pd.DataFrame: DataFrame actualizado con las métricas calculadas.
    """
    # Realizar predicciones
    y_pred = modelo.predict(X_test)

    # Calcular métricas básicas
    exactitud = round(accuracy_score(y_test, y_pred), 2)  # Porcentaje de predicciones correctas
    puntuaciones = cross_val_score(modelo, X_train, y_train, cv=5)  # Validación cruzada
    exactitud_CVS = "%0.2f (+/- %0.2f)" % (puntuaciones.mean(), puntuaciones.std() * 2)  # Promedio y rango
    precision_promedio = round(np.mean(precision_score(y_test, y_pred, average=None)), 2)  # Promedio de precisión
    recall_promedio = round(np.mean(recall_score(y_test, y_pred, average=None)), 2)  # Promedio de recall
    puntuacion_f1_promedio = round(np.mean(f1_score(y_test, y_pred, average=None)), 2)  # Promedio de F1
    cohen_kappa = round(cohen_kappa_score(y_test, y_pred), 2)  # Índice Cohen Kappa

    # Calcular ROC-AUC para problemas binarios
    if len(np.unique(y_test)) == 2:
        roc_auc = round(roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1]), 2)  # Usa probabilidades si están disponibles
    else:
        roc_auc = "N/A"  # No aplicable para multiclase sin binarización

    # Crear un diccionario con las métricas
    metrics = {
        "Nombre": nombre,
        "Transformación": transformacion,
        "Descripción": descripcion,
        "Exactitud": exactitud,
        "Exactitud_CVS": exactitud_CVS,
        "Precisión_Promedio": precision_promedio,
        "Recall_Promedio": recall_promedio,
        "Puntuación_F1_Promedio": puntuacion_f1_promedio,
        "Cohen_Kappa": cohen_kappa,
        "ROC_AUC": roc_auc
    }

    # Si no existe el DataFrame, inicializarlo
    if df is None:
        columnas = list(metrics.keys())
        df = pd.DataFrame(columns=columnas)

    # Agregar las métricas al DataFrame
    df = pd.concat([df, pd.DataFrame([metrics])], ignore_index=True)

    return df

# Interpretación de métricas de clasificación:
# -------------------------------------------
# 1. Exactitud (Accuracy):
#    - Rango: 0 a 1 (o 0% a 100%).
#    - Representa el porcentaje de predicciones correctas del modelo.
#    - Más alto es mejor. Un valor cercano a 1 indica que el modelo clasifica correctamente la mayoría de las instancias.
#    - Útil en datasets balanceados, pero puede ser engañoso si hay desbalance de clases.

# 2. Validación Cruzada (Exactitud_CVS):
#    - Proporciona el promedio y la desviación estándar de la exactitud en validación cruzada.
#    - Más alto es mejor, con menor desviación indicando mayor estabilidad.
#    - Útil para evaluar la generalización del modelo.

# 3. Precisión (Precision):
#    - Rango: 0 a 1 (o 0% a 100%).
#    - Define el porcentaje de predicciones positivas que son realmente correctas.
#    - Más alto es mejor. Valores bajos indican muchos falsos positivos.
#    - Importante en problemas donde los falsos positivos son costosos (ej., spam, diagnóstico de enfermedades).

# 4. Recall (Sensibilidad o Tasa de Verdaderos Positivos):
#    - Rango: 0 a 1 (o 0% a 100%).
#    - Mide el porcentaje de casos positivos reales que el modelo identifica correctamente.
#    - Más alto es mejor. Valores bajos indican muchos falsos negativos.
#    - Crítico en problemas donde los falsos negativos son costosos (ej., detección de fraude, enfermedades críticas).

# 5. F1 Score:
#    - Rango: 0 a 1 (o 0% a 100%).
#    - Promedio armónico entre precisión y recall.
#    - Más alto es mejor. Es especialmente útil para datasets desbalanceados.
#    - Penaliza los casos en los que precisión o recall son muy bajos.

# 6. Índice Cohen Kappa:
#    - Rango: -1 a 1.
#    - Mide el grado de acuerdo entre las predicciones del modelo y la realidad, ajustado por el azar.
#    - Interpretación:
#        - < 0: Sin acuerdo.
#        - 0 a 0.20: Acuerdo ligero.
#        - 0.21 a 0.40: Acuerdo aceptable.
#        - 0.41 a 0.60: Acuerdo moderado.
#        - 0.61 a 0.80: Acuerdo considerable.
#        - 0.81 a 1.00: Acuerdo casi perfecto.
#    - Más alto es mejor.

# 7. Área Bajo la Curva ROC (ROC-AUC):
#    - Rango: 0.5 a 1.
#    - Mide la capacidad del modelo para distinguir entre clases positivas y negativas.
#    - Interpretación:
#        - 0.5: Rendimiento aleatorio (el modelo no tiene capacidad discriminatoria).
#        - 0.7 a 0.8: Rendimiento aceptable.
#        - 0.8 a 0.9: Rendimiento excelente.
#        - > 0.9: Rendimiento excepcional.
#    - Más alto es mejor. Un valor cercano a 1 indica que el modelo distingue perfectamente entre clases.

# Notas adicionales:
# - Las métricas ideales dependen del problema y del contexto.
# - Para datasets desbalanceados, F1 Score, ROC-AUC y recall son más informativas que la exactitud.