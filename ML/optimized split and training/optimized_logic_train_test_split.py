# Importaciones necesarias
import os
import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Función para cargar tensores de audio
def cargar_tensor_de_audio_desde_ruta(ruta_archivo):
    """
    Carga un tensor de audio desde una ruta de archivo específica.

    Args:
        ruta_archivo (str): Ruta del archivo de audio.

    Returns:
        tensor: El tensor de audio cargado o None si ocurre un error.
    """
    try:
        tensor = torch.load(ruta_archivo)
        return tensor
    except Exception as e:
        print(f"Error al cargar el tensor desde {ruta_archivo}: {str(e)}")
        return None

# Directorio base para guardar resultados
directorio_base = "ruta/a/tu/directorio/base"

# Lista de combinaciones de características
combinaciones_caracteristicas = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6']

# Tipos de características
tipos_caracteristicas = ['type_a', 'type_b']

# Función principal
def procesar_datos(dataset, directorio_base, combinaciones_caracteristicas, tipos_caracteristicas):
    """
    Procesa datos de audio para diferentes combinaciones de características y guarda
    los resultados en archivos pickle tras realizar LDA y división train-test.

    Args:
        dataset (pd.DataFrame): Dataset etiquetado con columnas como 'filepath' y características objetivo.
        directorio_base (str): Directorio base donde se guardarán los resultados.
        combinaciones_caracteristicas (list): Lista de características objetivo.
        tipos_caracteristicas (list): Lista de tipos de características (ej. type_a, type_b).
    """
    for caracteristica in combinaciones_caracteristicas:
        for tipo_caracteristica in tipos_caracteristicas:
            # Crear ruta para guardar resultados
            directorio_caracteristica = os.path.join(directorio_base, caracteristica, tipo_caracteristica)
            os.makedirs(directorio_caracteristica, exist_ok=True)

            # Filtrar dataset por tipo de característica
            if tipo_caracteristica == 'type_a':
                rutas_archivos = dataset[dataset['subfolder'] == 'type_a']['filepath']
            elif tipo_caracteristica == 'type_b':
                rutas_archivos = dataset[dataset['subfolder'] == 'type_b']['filepath']

            # Cargar tensores de audio
            tensores_audio = [cargar_tensor_de_audio_desde_ruta(ruta) for ruta in rutas_archivos if ruta]
            tensores_audio = [tensor for tensor in tensores_audio if tensor is not None]

            # Convertir lista de tensores en un tensor de PyTorch
            X = torch.stack(tensores_audio).view(len(tensores_audio), -1)

            # Definir variable objetivo (y)
            if caracteristica in ['feature_5', 'feature_6']:
                y = dataset[dataset['subfolder'] == tipo_caracteristica][caracteristica]
            else:
                y = dataset[dataset['subfolder'] == tipo_caracteristica][f"{caracteristica}_encoded"]

            # Reducir dimensiones con LDA
            n_componentes_lda = min(X.shape[1], len(np.unique(y)) - 1)
            n_componentes_lda = min(n_componentes_lda, 7)  # Máximo de 7 componentes
            lda = LinearDiscriminantAnalysis(n_components=n_componentes_lda)
            X_lda = lda.fit_transform(X, y)

            # Guardar proyecciones LDA
            with open(os.path.join(directorio_caracteristica, 'X_lda.pkl'), 'wb') as f:
                pickle.dump(X_lda, f)

            # Dividir datos en entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

            # Guardar conjuntos de datos
            with open(os.path.join(directorio_caracteristica, 'X_train.pkl'), 'wb') as f:
                pickle.dump(X_train, f)
            with open(os.path.join(directorio_caracteristica, 'X_test.pkl'), 'wb') as f:
                pickle.dump(X_test, f)
            with open(os.path.join(directorio_caracteristica, 'y_train.pkl'), 'wb') as f:
                pickle.dump(y_train, f)
            with open(os.path.join(directorio_caracteristica, 'y_test.pkl'), 'wb') as f:
                pickle.dump(y_test, f)

            # Guardar datos originales
            with open(os.path.join(directorio_caracteristica, 'X.pkl'), 'wb') as f:
                pickle.dump(X, f)
            with open(os.path.join(directorio_caracteristica, 'y.pkl'), 'wb') as f:
                pickle.dump(y, f)

# # Ejemplo de uso
# if __name__ == "__main__":
#     """
#     Este bloque demuestra cómo utilizar la función procesar_datos para preparar los datos
#     y guardarlos en diferentes combinaciones de características y tipos.
#     """
#     # Cargar dataset (reemplazar con la ruta correcta del archivo pickle)
#     with open("ruta/a/tu/dataset_labeled.pkl", 'rb') as archivo:
#         dataset = pickle.load(archivo)

#     # Procesar datos
#     procesar_datos(dataset, directorio_base, combinaciones_caracteristicas, tipos_caracteristicas)
