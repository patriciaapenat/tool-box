# Importaciones necesarias
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import (
    NoSuchElementException,
    WebDriverException,
)
import time
import random
import pandas as pd
import os

# Configuración general
# ---------------------
# Ruta del controlador de Chrome (modificar según tu sistema)
chrome_driver_path = r"path/to/your/chromedriver"

# Carpeta de salida y nombre del archivo CSV donde se guardarán los datos extraídos
output_folder = r"path/to/your/output/folder"
csv_filename = os.path.join(output_folder, "scraped_data.csv")

# Inicializamos una lista vacía para almacenar los datos extraídos
data = []

# Inicialización del controlador de Selenium
# ------------------------------------------
# Configura y lanza el navegador Chrome usando Selenium.
def initialize_driver():
    """
    Inicializa el controlador de Chrome con opciones configuradas para evitar detección y mejorar rendimiento.
    """
    global driver, service, options
    try:
        service = ChromeService(executable_path=chrome_driver_path)
        options = ChromeOptions()
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--headless")  # Ejecutar en modo headless (sin mostrar la ventana)
        driver = webdriver.Chrome(service=service, options=options)
        
        # Ocultar la propiedad 'webdriver' para evitar detección por algunos sitios
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        })
    except WebDriverException as e:
        print(f"Error al inicializar el controlador de Chrome: {str(e)}")
        exit()

# Funciones auxiliares
# --------------------
# Estas funciones proporcionan funcionalidades básicas para gestionar la interacción con la página.

def random_sleep():
    """
    Pausa la ejecución por un tiempo aleatorio para simular el comportamiento humano.
    """
    time.sleep(random.uniform(1, 5))

def accept_cookies():
    """
    Intenta aceptar cookies si la ventana de cookies está presente.
    """
    try:
        cookies_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[1]/div[2]/span[1]/a')
        cookies_button.click()
        random_sleep()
        print("Cookies aceptadas.")
    except NoSuchElementException:
        print("No se encontraron cookies para aceptar.")

def save_data_to_csv(entry):
    """
    Guarda los datos extraídos en un archivo CSV.
    Si el archivo no existe, lo crea con encabezados; si ya existe, añade datos como nuevas filas.
    """
    df = pd.DataFrame([entry])
    if not os.path.exists(csv_filename):
        df.to_csv(csv_filename, index=False, mode="w", header=True, encoding="utf-8-sig")
    else:
        df.to_csv(csv_filename, index=False, mode="a", header=False, encoding="utf-8-sig")
    print(f"Entrada guardada: {entry['Index']} - {entry['Name']}")

# Extracción de información de una empresa
# -----------------------------------------
# Esta función asume que la página de detalles de una empresa ya está cargada.
def scrape_company_info(url, data, index):
    """
    Extrae y guarda la información de una empresa en la lista de datos.
    
    Args:
        url (str): URL de la página de la empresa.
        data (list): Lista donde se almacenarán los datos extraídos.
        index (int): Índice único que identifica a la empresa.
    """
    try:
        name = driver.find_element(By.CSS_SELECTOR, 'h1').text
        place = driver.find_element(By.CSS_SELECTOR, 'address span:nth-child(3)').text
        category = driver.find_element(By.CSS_SELECTOR, 'section:nth-child(2) div:nth-child(2) span a').text
        web = driver.find_element(By.CSS_SELECTOR, 'section:nth-child(4) ul:nth-child(2) li:nth-child(2) a').get_attribute('href')
        address = driver.find_element(By.CSS_SELECTOR, 'section:nth-child(2) address').text.replace('\n', '. ')
        
        entry = {
            "Index": index,
            "Name": name,
            "Place": place,
            "Category": category,
            "Web": web,
            "Address": address,
            "Comes From": url
        }
        data.append(entry)
        save_data_to_csv(entry)
    except NoSuchElementException as e:
        print(f"Error al extraer información de la empresa: {e}")

# Gestión de la página principal y scraping
# ------------------------------------------
def scrape_page(url):
    """
    Accede a la página principal e interactúa con ella para cargar los datos necesarios.

    Args:
        url (str): URL de la página principal de búsqueda.
    """
    try:
        driver.get(url)
        accept_cookies()
        random_sleep()
        print("Página principal cargada correctamente.")
    except WebDriverException as e:
        print(f"Error al acceder a la página principal: {e}")
        driver.quit()
        exit()

# URL pública para scraping
# -------------------------
# URL del sitio web público donde se realizará el scraping.
base_url = "https://www.dastelefonbuch.de/Suche/Schwimmbadbau?s=eyJvcmRlcmJ5IjoibmFtZSJ9"

# Flujo principal
# ----------------
# Este flujo inicializa el controlador, accede a la página principal y cierra el navegador al finalizar.
if __name__ == "__main__":
    initialize_driver()  # Inicia el controlador de Chrome
    try:
        scrape_page(base_url)  # Accede a la página principal
        # Aquí se pueden implementar más funciones para iterar y extraer información.
    finally:
        driver.quit()  # Cierra el navegador al final
        print("Proceso de scraping finalizado.")