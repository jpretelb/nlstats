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

from fetch_data import local_connection
from bs4 import BeautifulSoup
from constants import clean_name, PATH_KB

import chromadb


from constants import PATH_CHROMA_DB, COLLECTION_NAME_PROBLEM


LOTE_EMBEDDING = 10

CARPETA_SALIDA = "incidencias"

from prompts import crear_prompt_consulta_resumen, crear_prompt_thema








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
                
                prompt_final = crear_prompt_consulta_resumen(texto_completo)
            

                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt_final
                    )

                    json_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                    datos_limpios = json.loads(json_text)
                    datos_limpios['idincidencia'] = idincidencia
                    datos_limpios['empresa'] = empresa
                    datos_limpios['modulo'] = modulo
                    datos_limpios['fechacreacion'] = fechacreacion.strftime("%Y-%m-%d")
                    
                    #print("4")

                    with open(ruta_archivo_salida, 'w', encoding='utf-8') as outfile:
                        json.dump(datos_limpios, outfile, ensure_ascii=False, indent=4)

                except Exception as e:
                    print(f"\nError al procesar la incidencia {idincidencia} con la API: {e}")
                    ids_con_error.append(idincidencia)

    except pyodbc.Error as ex:
        print(f"Error al conectar a la base de datos: {ex}")
        return
    
    
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

