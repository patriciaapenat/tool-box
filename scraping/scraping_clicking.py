# Importaciones necesarias
import pandas as pd
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.common.exceptions import (
    NoSuchElementException,
    TimeoutException,
    StaleElementReferenceException,
    WebDriverException,
    ElementClickInterceptedException,
)
import time
import random

# Configuración general del proyecto
# ----------------------------------
# Ruta del controlador de Chrome (modifica según tu sistema)
chrome_driver_path = r"path/to/your/chromedriver"

# Carpeta para guardar los archivos CSV (modifica según tu entorno)
folder_path = r"path/to/your/output/folder"

# Verificar archivos CSV existentes para evitar procesar regiones duplicadas
existing_files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('.csv')]

# Funciones de configuración
# --------------------------
def initialize_driver():
    """
    Inicializa el controlador de Selenium (Chrome) con opciones configuradas.
    """
    global driver, wait
    service = ChromeService(executable_path=chrome_driver_path)
    options = ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-extensions")
    options.add_argument("--disable-gpu")
    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 3)

    # Ocultar la propiedad "webdriver" para evadir detecciones
    driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
        "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    })

# Inicializa el controlador
initialize_driver()

# Funciones auxiliares
# --------------------
def random_sleep():
    """
    Pausa la ejecución por un tiempo aleatorio para simular comportamiento humano.
    """
    time.sleep(random.uniform(1, 2))

def click_element_safely(element):
    """
    Intenta hacer clic en un elemento utilizando diferentes estrategias para evitar errores.
    """
    try:
        driver.execute_script("arguments[0].scrollIntoView(true);", element)
        random_sleep()
        element.click()
    except ElementClickInterceptedException:
        print("El clic fue interceptado. Intentando clic con JavaScript.")
        driver.execute_script("arguments[0].click();", element)
    except Exception as e:
        print(f"Error inesperado al hacer clic: {e}")

def extract_text_by_selector(selector, attribute=None):
    """
    Extrae texto o atributo de un elemento encontrado por su selector CSS.
    """
    try:
        element = driver.find_element(By.CSS_SELECTOR, selector)
        return element.get_attribute(attribute) if attribute else element.text
    except NoSuchElementException:
        return ""

def process_name(name):
    """
    Limpia el nombre eliminando caracteres innecesarios como '\n' o '\'.
    """
    if '\\' in name:
        return name.split('\\')[0].strip()
    if '\n' in name:
        return name.split('\n')[0].strip()
    return name.strip()

def process_place_from_location(location):
    """
    Extrae el lugar de la ubicación eliminando etiquetas como "Piscine à ".
    """
    return location.replace("Piscine à ", "").strip()

# Función de scraping
# -------------------
def scrape_company_info(url, data):
    """
    Extrae y guarda información detallada de una empresa desde la página de su perfil.
    
    Args:
        url (str): URL de la empresa.
        data (list): Lista para almacenar los datos extraídos.
    """
    try:
        driver.get(url)
        random_sleep()

        # Extraer campos básicos de la empresa
        name = process_name(extract_text_by_selector('h1'))
        phone = extract_text_by_selector('#coord-liste-numero_1 > span > span.coord-numero-mobile.noTrad > a', attribute='href')
        if phone and phone.startswith('tel:'):
            phone = phone[4:]
        category = extract_text_by_selector('#teaser-header .zone-activites span')
        address = extract_text_by_selector('#teaser-footer .address.streetAddress') or "No disponible"
        place = process_place_from_location(extract_text_by_selector('#blocMaillage > div > a:nth-child(3)'))

        # Extraer sitios web
        web_elements = driver.find_elements(By.XPATH, "//div[@class='bloc-info-sites-reseaux']//a")
        web = ' / '.join([link.text for link in web_elements])

        # Imprimir los resultados para verificar
        print(f"Name: {name}, Phone: {phone}, Category: {category}, Address: {address}, Place: {place}, Web: {web}")

        # Agregar datos a la lista
        data.append({
            "Name": name,
            "Phone": phone,
            "Category": category,
            "Address": address,
            "Place": place,
            "Web": web,
            "URL": url
        })
    except Exception as e:
        print(f"Error al extraer información de la empresa: {e}")

def scrape_region(region_name, region_formats):
    """
    Realiza el scraping de una región específica iterando sobre sus variaciones de formato.
    
    Args:
        region_name (str): Nombre de la región.
        region_formats (list): Lista de variaciones en el formato del nombre.
    """
    data = []
    for region_format in region_formats:
        page = 1
        while True:
            page_url = f"https://www.pagesjaunes.fr/annuaire/chercherlespros?quoiqui=construction%20piscine&ou={region_format}&page={page}"
            print(f"Scraping URL: {page_url}")
            try:
                driver.get(page_url)
                random_sleep()

                # Lista de empresas en la página
                company_list = driver.find_elements(By.XPATH, '//main/div[2]/div/section/div/ul/li')
                if not company_list:
                    break  # Terminar si no hay más resultados

                for index, company in enumerate(company_list, start=1):
                    try:
                        company_element = driver.find_element(By.XPATH, f'//main/div[2]/div/section/div/ul/li[{index}]/div[1]/div/div/div[1]/a/h3')
                        print(f"Processing: {company_element.text}")
                        click_element_safely(company_element)
                        scrape_company_info(driver.current_url, data)
                        driver.back()
                        random_sleep()
                    except NoSuchElementException:
                        continue

                page += 1  # Ir a la siguiente página
            except WebDriverException as e:
                print(f"Error al procesar la URL {page_url}: {e}")
                break

    # Guardar resultados en un archivo CSV
    if data:
        df = pd.DataFrame(data)
        csv_file = os.path.join(folder_path, f"{region_name}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Datos guardados en {csv_file}")

# Regiones a procesar
regions = [
    "Martinique", "Guadeloupe", "Guyane", "La Reunion"
]

# Procesar regiones
for region in regions:
    region_name = region.replace(' ', '-')
    if region_name in existing_files:
        print(f"Región {region_name} ya procesada. Saltando.")
        continue
    region_formats = [region.replace(' ', '+'), region.replace(' ', '-')]
    scrape_region(region_name, region_formats)

# Cerrar el navegador al final
driver.quit()