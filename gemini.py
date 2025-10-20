import os
import json
import pyodbc
#import google.generativeai as genai
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

from dotenv import load_dotenv
from tqdm import tqdm
import math

from step0_migration import local_connection
from bs4 import BeautifulSoup
from constants import clean_name, PATH_KB

import chromadb


from constants import PATH_CHROMA_DB, COLLECTION_NAME_PROBLEM


LOTE_EMBEDDING = 10

CARPETA_SALIDA = "incidencias"

def crear_prompt(texto_completo_incidencia):
    """Crea el prompt estructurado para la IA."""

    instruccion = """## ROL Y OBJETIVO
Eres un Analista de Datos experto en Procesamiento de Lenguaje Natural (NLP). Tu misión es procesar incidencias de soporte del software NISIRA ERP para crear un corpus de texto limpio, consistente y estandarizado para análisis estadístico. La consistencia y la creación de tokens únicos para conceptos clave y en singular (estrategia "single word + singluar") es el objetivo principal.

---
## TAREA PRINCIPAL
Analiza la siguiente incidencia, extrae la información clave y reescríbela de forma normalizada en un JSON estructurado, aplicando rigurosamente las siguientes reglas.

---
## REGLAS DE NORMALIZACIÓN
1.  **ESTRATEGIA SINGLE WORD (CRÍTICO):** Normaliza siempre los nombres completos y sus abreviaturas a la sigla o token único oficial. El objetivo es que cada concepto sea una sola palabra. Por ejemplo:
    * "Orden de Servicio" -> **"OSC"**
    * "Orden de Compra" -> **"OCO"**
    * "Guía de remisión" -> **"GRM"**
    * "Nota de credito" -> **"NCR"**
    * "Compensación por tiempo de servicio" -> **"CTS"**
    * "Liquidacion de Beneficios sociales" o similar -> **"Liquidaciones"**
    * "Orden de Venta" -> **"OVT"**
    * "Renta de quinta categoria" o "renta 5ta" -> **"R5TA"**
    * "Recursos Humanos" o "Nómina" -> **"RRHH"**
    * "Packing List" -> **"Packing"**
    * "Packing List" -> **"Packing"**
    * "Packing List" -> **"Packing"**
    * "Cuentas por Pagar", "Cuentas/pagar" -> **"Cuentas_por_Pagar"**
    * "Cuentas por Cobrar", "Cuentas-x-Cobrar" -> **"Cuentas_por_Cobrar"**
    * "Base de Datos" -> **"BD"**

2.  **FORMA SINGULAR (CRÍTICO):** Convierte siempre los sustantivos plurales a su forma singular. El objetivo es unificar el mismo concepto en una sola palabra, puedes colocar entre parentesis la palabra varios.
    * **Ejemplo:** "Las facturas no coinciden" -> se convierte en "La factura no coincide (varios)".
    * **Ejemplo:** "Errores en las órdenes de compra" -> se convierte en "Error en la orden de compra (varios)".
    * **Ejemplo:** "Formulario de liquidaciones" -> se convierte en "Formulario de liquidacion".
    * **Ejemplo:** "Documento de liquidaciones" -> se convierte en "Documento de liquidacion".

3.  **FECHAS Y PERIODOS:** Normaliza todas las fechas y periodos al formato **mes año** (ej: `enero 2024`) o **dd mes año** (ej: `31 enero 2024`). Usa siempre el nombre del mes completo y en minúsculas.
    * **Ejemplo:** "31/01/2024" -> **"31 enero 2024"**
    * **Ejemplo:** "enero 2024", "01-2024", "202401" -> **"enero 2024"**
    * **Ejemplo:** "periodo 2024-2025" -> **"periodo 2024 2025"**


4.  **ANONIMIZACIÓN:**
    * Reemplaza nombres de personas por la palabra **"el trabajador"** (solo si es relevante).
    * Reemplaza nombres o siglas de empresas clientes (ej: "GMHBERRIES", "GMH") por la frase **"la empresa"**.

5.  **LIMPIEZA GENERAL:**
    * Elimina frases de cortesía ("hola", "gracias"), información personal (teléfonos, emails), caracteres especiales, comillas y saltos de línea.
    * Reescribe el problema y la solución de forma clara y concisa.



---
## FORMATO DE SALIDA (JSON)
El resultado debe ser un JSON estricto con estas claves. Si no hay solución, el valor debe ser `null`.
* `problema`: (string) Descripción normalizada.
* `solucion`: (string|null) Descripción normalizada.
* `categoria`: (string) La categoría más apropiada.
* `urgencia_estimada`: (string) "Alta", "Media" o "Baja".
* `entidades`: (array de objetos) Cada objeto con `{"tipo": "...", "nombre": "..."}`.

---
## CONTEXTO Y DICCIONARIO
* **ERP:** NISIRA ERP
* **Módulos Principales:** Ventas, Compras, Almacén, Contabilidad, Activo Fijo, Bancos, Cuentas_por_Cobrar, Cuentas_por_Pagar, RRHH, Producción, Mantenimiento.
* **Siglas Principales:** PLE, SIRE, PLAME, EDOC, SUNAT.
* **Tipos de Entidades:** "Módulo", "Proceso", "Documento", "Entidad Bancaria", "Concepto Contable", "Entidad Fiscal".

---
## EJEMPLO PRÁCTICO (FEW-SHOT)

### INPUT (Texto de la incidencia):
"hola amigos de GMHBERRIES, tengo un problm con la renta de 5ta cat. para el empleado Juan Perez en la Planilla Mensual de Pagos de enero 2024, el calculo del impuesto no me cuadra con el reporte del nisira erp, además tengo problemas con el calulo de liquidaciones y otros beneficios sociales. Me sale error."

### OUTPUT (JSON Esperado):
```json
{
    "problema": "El cálculo del impuesto R5TA para el trabajador en PLAME del periodo enero 2024 es incorrecto o no coincide con los montos del reporte generado por el sistema NISIRA ERP en la empresa. Y tiene problema con el cálculo de la liquidacion (varios) y beneficio social.",
    "solucion": null,
    "categoria": "Cálculo de Nómina / RRHH",
    "urgencia_estimada": "Alta",
    "entidades": [
        {
            "tipo": "Módulo",
            "nombre": "RRHH"
        },
        {
            "tipo": "Concepto Contable",
            "nombre": "R5TA"
        },
        {
            "tipo": "Documento",
            "nombre": "PLAME"
        }
    ]
}"""
    
    prompt = f"{instruccion}\n\nConversación de la Incidencia:\n---\n{texto_completo_incidencia}\n---\n\nJSON de Salida:"
    return prompt




def crear_prompt_thema(texto_completo_incidencia):
    
    instruccion = """## ROL Y OBJETIVO
Eres un Analista de Datos experto en Procesamiento de Lenguaje Natural (NLP). Tu misión es procesar incidencias de soporte 
del software NISIRA ERP para crear temas por cada cluster que te envio.

---
## TAREA PRINCIPAL
Analiza cada cluster y colocale un titulo o tema representativo.

## REDACCION
La redaccion debe ser corta, evitar el uso de emojis.

## TERMINOS CLAVE
Estos terminos fueron reducidos, pero debes tener en cuenta las siguientes equivalencias para una mejor interpretacion:
R5TA : Renta de quita categoria

## FORMATO DE SALIDA (JSON)
El resultado debe ser un JSON estricto con estas claves.
* `cluster_id`: (integer) Número de cluster.
* `tema`: (string) Thema encontrado.

### OUTPUT (JSON Esperado):
```json
    [
        {"cluster_id": 0, "tema": "Errores en Cálculo de Impuestos"},
        {"cluster_id": 1, "tema": "Problemas con Reportes Financieros"},
        {"cluster_id": 2, "tema": "Dificultades en Gestión de Inventarios"}
    ]```"""
    
    prompt = f"{instruccion}\n\nClusters encontrados:\n---\n{texto_completo_incidencia}\n---\n\nJSON de Salida:"
    return prompt



def return_etiquetas_from_cluster(texto_completo):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key de Gemini no encontrada. Revisa tu archivo .env.")
    client = genai.Client(api_key=api_key)
    model = 'gemini-2.0-flash' 
    
    
    prompt_final = crear_prompt_thema(texto_completo)
    
    response = client.models.generate_content(
        model=model,
        contents=prompt_final
    )
    try:     
        response = client.models.generate_content(
            model=model,
            contents=prompt_final
        )
        #print(response.text)
        #print("1")
        json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        #print("2")
        #json_text = response.text.strip()
        datos_limpios = json.loads(json_text)
        
        
        return datos_limpios
        
    except Exception as e:
        print(f"\nError al procesar etiquetas con la API: {e}")
        return

def run_normaliza_gemini(postgres_conf, postgres_pass, year_month):
    """
    Script principal que procesa incidencias y guarda cada una en un archivo JSON,
    verificando si ya existe para evitar reprocesamiento.
    """
    # --- 1. CONFIGURACIÓN ---
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key de Gemini no encontrada. Revisa tu archivo .env.")
    
    client = genai.Client(api_key=api_key)
    model = 'gemini-2.0-flash'

    if not os.path.exists(CARPETA_SALIDA):
        os.makedirs(CARPETA_SALIDA)
        print(f"Carpeta de salida '{CARPETA_SALIDA}' creada.")

    ids_con_error = []
    incidencias_saltadas = 0

    
    try:
        with local_connection(postgres_conf['server'], postgres_conf['port'], postgres_conf['database'], postgres_conf['username'], postgres_pass) as conn:
            cursor = conn.cursor()
            
            print("Obteniendo incidencias principales desde la base de datos...")
            query = f"SELECT id, titulo, contenido, empresa, fechacreacion, modulo FROM incidencia where TO_CHAR(fechacreacion, 'YYYYMM') = '{year_month}'"
            cursor.execute(query)
            incidencias_principales = cursor.fetchall()
            print(f"Se encontraron {len(incidencias_principales)} incidencias. Iniciando procesamiento...")

            for incidencia_row in tqdm(incidencias_principales, desc="Procesando incidencias"):
                idincidencia, titulo, contenido, empresa, fechacreacion, modulo = incidencia_row
                
                # --- NUEVO: Lógica de verificación ---
                ruta_archivo_salida = os.path.join(CARPETA_SALIDA, f"{idincidencia}.json")
                if os.path.exists(ruta_archivo_salida):
                    incidencias_saltadas += 1
                    continue # Salta a la siguiente iteración del bucle si el archivo ya existe
                
                # Si el archivo no existe, continúa con el proceso
                soup = BeautifulSoup(contenido, 'html.parser')
                contenido_parser = soup.get_text(separator=' ', strip=True)

                texto_completo = f"TÍTULO: {titulo}\nREPORTE INICIAL: {contenido_parser}\n\n--- HISTORIAL DE RESPUESTAS ---\n"

                query = f"SELECT tipousuario as autor, respuesta FROM respuesta WHERE idincidencia = {idincidencia} ORDER BY fechacreacion ASC"

                cursor.execute(query)
                respuestas = cursor.fetchall()
                
                for respuesta_item in respuestas:
                    autor, respuesta_texto = respuesta_item
                    soup = BeautifulSoup(respuesta_texto, 'html.parser')
                    respuesta_parser = soup.get_text(separator=' ', strip=True)

                    texto_completo += f"\n{autor}: {respuesta_parser}\n"
                
                prompt_final = crear_prompt(texto_completo)
            

                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt_final
                    )

                    #print(response.text)
                    #print("1")
                    json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    #print("2")
                    #json_text = response.text.strip()
                    datos_limpios = json.loads(json_text)
                    #print("3")
                    datos_limpios['idincidencia'] = idincidencia
                    datos_limpios['empresa'] = empresa
                    datos_limpios['modulo'] = modulo
                    datos_limpios['fechacreacion'] = fechacreacion.strftime("%Y-%m-%d")
                    
                    #print("4")

                    # --- NUEVO: Guardado individual ---
                    with open(ruta_archivo_salida, 'w', encoding='utf-8') as outfile:
                        json.dump(datos_limpios, outfile, ensure_ascii=False, indent=4)

                except Exception as e:
                    print(f"\nError al procesar la incidencia {idincidencia} con la API: {e}")
                    ids_con_error.append(idincidencia)

    except pyodbc.Error as ex:
        print(f"Error al conectar a la base de datos: {ex}")
        return
    
    



    

    # --- 3. REPORTE FINAL ---
    print("\n--- REPORTE DEL PROCESO ---")
    total_procesadas = len(incidencias_principales) - incidencias_saltadas - len(ids_con_error)
    print(f"Incidencias procesadas en esta ejecución: {total_procesadas}")
    print(f"Incidencias saltadas (ya existían): {incidencias_saltadas}")
    print(f"Incidencias con error: {len(ids_con_error)}")

    if ids_con_error:
        with open('incidencias_con_error.txt', 'w', encoding='utf-8') as error_file:
            for item_id in ids_con_error:
                error_file.write(f"{item_id}\n")
        print(f"Se guardaron los IDs con error en 'incidencias_con_error.txt'.")

    print("\nProceso completado.")




def rag(module):

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key de Gemini no encontrada. Revisa tu archivo .env.")


    clean_name_module = clean_name(module)
    path = PATH_KB + "/" + clean_name_module

    archivos_md = [f for f in os.listdir(path) if f.endswith('.md')]
    print(f"Cargando documentos desde '{path}'...")

    documentos = []
    metadatos = []
    ids = []


    for filename in tqdm(archivos_md, desc="Cargando archivos .md"):
        filepath = os.path.join(path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            documentos.append(f.read())
            # Guardamos el nombre del cluster como metadato
            metadatos.append({'grupo': clean_name_module, 'cluster_name': filename.replace('.md', '')})
            ids.append(clean_name_module + "_" + filename)
    
    if not documentos:
        print("Error: No se encontraron documentos .md. Ejecuta el script de consolidación primero.")
        exit()
    
    print(f"{len(documentos)} documentos cargados.")

    print("Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=PATH_CHROMA_DB)

    embedding_function = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)

    print(f"Creando o cargando la colección '{COLLECTION_NAME_PROBLEM}'...")
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME_PROBLEM,
        embedding_function=embedding_function
    )

    filtro = {"grupo": clean_name_module}

    #registros_a_eliminar = collection.count(where=filtro)
    #    
    #print(f"**Se encontraron {registros_a_eliminar} registros para el módulo '{clean_name_module}'**.")
    #
    #if registros_a_eliminar > 0:#

    #    print(f"\n**Limpiando documentos del módulo/cluster: '{clean_name_module}'**")
    #    
    #    collection.delete(#
    #        where={"grupo": clean_name_module} 
    #    )
    #    print(f"Limpieza de '{clean_name_module}' completada.")

    print(f"\n**Limpiando documentos del módulo/cluster: '{clean_name_module}'**")

    collection.delete(
        where={"grupo": clean_name_module} 
    )

    print(f"Limpieza de '{clean_name_module}' completada.")
    
    print("Añadiendo documentos a la colección (esto puede tardar)...")

    num_lotes = math.ceil(len(documentos) / LOTE_EMBEDDING)

    for i in tqdm(range(num_lotes), desc="Procesando y guardando embeddings"):
        inicio = i * LOTE_EMBEDDING
        fin = (i + 1) * LOTE_EMBEDDING
        
        lote_docs = documentos[inicio:fin]
        lote_metadatos = metadatos[inicio:fin]
        lote_ids = ids[inicio:fin]
        
        collection.add(
            documents=lote_docs,
            metadatas=lote_metadatos,
            ids=lote_ids
        )

def rag_all(path, chroma_db_path, collection_name):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API Key de Gemini no encontrada. Revisa tu archivo .env.")

    archivos_md = [f for f in os.listdir(path) if f.endswith('.md')]
    print(f"Cargando documentos desde '{path}'...")

    documentos = []
    metadatos = []
    ids = []

    for filename in tqdm(archivos_md, desc="Cargando archivos .md"):
        filepath = os.path.join(path, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            documentos.append(f.read())
            # Guardamos el nombre del cluster como metadato
            metadatos.append({'cluster_name': filename.replace('.md', '')})
            ids.append(filename)
    
    if not documentos:
        print("Error: No se encontraron documentos .md. Ejecuta el script de consolidación primero.")
        exit()
    print(f"{len(documentos)} documentos cargados.")

    print("Inicializando ChromaDB...")
    client = chromadb.PersistentClient(path=chroma_db_path)

    embedding_function = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)

    print(f"Creando o cargando la colección '{collection_name}'...")
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_function
    )

    print("Añadiendo documentos a la colección (esto puede tardar)...")

    num_lotes = math.ceil(len(documentos) / LOTE_EMBEDDING)

    for i in tqdm(range(num_lotes), desc="Procesando y guardando embeddings"):
        inicio = i * LOTE_EMBEDDING
        fin = (i + 1) * LOTE_EMBEDDING
        
        lote_docs = documentos[inicio:fin]
        lote_metadatos = metadatos[inicio:fin]
        lote_ids = ids[inicio:fin]
        
        collection.add(
            documents=lote_docs,
            metadatas=lote_metadatos,
            ids=lote_ids
        )

