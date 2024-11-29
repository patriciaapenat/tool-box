import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler, Imputer, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml import Pipeline

# Si hay un SparkContext existente, debemos cerrarlo antes de crear uno nuevo
if 'sc' in locals() and sc:
    sc.stop()  # Detener el SparkContext anterior si existe

# Configuración de Spark
conf = (
    SparkConf()
    .setAppName("Nombre de la app")  # Nombre de la aplicación en Spark
    .setMaster("local[2]")  # Modo local con un hilo para ejecución
    .set("spark.driver.host", "127.0.0.1")  # Dirección del host del driver (este Localhost)
    .set("spark.executor.heartbeatInterval", "3600s")  # Intervalo de latido del executor
    .set("spark.network.timeout", "7200s")  # Tiempo de espera de la red
    .set("spark.executor.memory", "14g")  # Memoria asignada para cada executor
    .set("spark.driver.memory", "14g")  # Memoria asignada para el driver
)

# Crear un nuevo SparkContext con la configuración especificada
sc = SparkContext(conf=conf)

# Configuración de SparkSession (interfaz de alto nivel para trabajar con datos estructurados en Spark)
spark = (
    SparkSession.builder
    .appName("Proyecto_PatriciaA_Peña")  # Nombre de la aplicación en Spark
    .config("spark.sql.repl.eagerEval.enabled", True)  # Habilitar la evaluación perezosa en Spark SQL REPL
    .config("spark.sql.repl.eagerEval.maxNumRows", 1000)  # Número máximo de filas a mostrar en la evaluación perezosa
    .getOrCreate()  # Obtener la sesión Spark existente o crear una nueva si no existe
) 