
import os
import shutil
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer

from constants import PATH_KB, clean_name, CSV_CONSOLIDATE, PATH_CLUSTER

INCIDENCIAS_POR_ARCHIVO = 20
TOP_N_LEMAS = 15

def generar_kb(module, lemma_column) :

    clean_name_module = clean_name(module)
    path = PATH_KB + "/" + clean_name_module
    csv_clusters = PATH_CLUSTER + "/" + clean_name_module + ".csv"




    if os.path.exists(path):
        print(f"Limpiando el directorio de salida existente: '{path}'...")
    else :
        os.makedirs(path, exist_ok=True)
    try:
        shutil.rmtree(path)
    except OSError as e:
        print(f"Error al eliminar el directorio {e.filename}: {e.strerror}")
        print("Por favor, elimina la carpeta manualmente y vuelve a intentarlo.")
        exit()

    os.makedirs(path)
    print(f"Directorio '{path}' creado y listo.")



    try:
        df_original = pd.read_csv(CSV_CONSOLIDATE)
        df_clusters = pd.read_csv(csv_clusters)
        df = pd.merge(df_clusters, df_original[['idincidencia', 'problema', 'solucion']], on='idincidencia')
        print(f"{len(df)} registros cargados y fusionados.")
    except FileNotFoundError as e:
        print(f"Error: No se encontró el archivo {e.filename}. Asegúrate de que los archivos de entrada existan.")
        exit()

    clusters_unicos = df['cluster_consolidado'].unique()
    print(f"\nSe encontraron {len(clusters_unicos)} clusters únicos. Generando archivos...")

    total_archivos_creados = 0

    for cluster in clusters_unicos:
        df_filtrado = df[df['cluster_consolidado'] == cluster].copy()
        num_incidencias_cluster = len(df_filtrado)
        
        if num_incidencias_cluster == 0:
            continue
        
        print("#2....")
        # Extraer lemas clave del cluster (se hace una vez por cluster)
        try:
            vectorizer = CountVectorizer(max_features=TOP_N_LEMAS)
            vectorizer.fit_transform(df_filtrado[lemma_column].dropna())
            lemas_clave = vectorizer.get_feature_names_out()
        except ValueError:
            lemas_clave = ["N/A (No hay suficientes datos para extraer lemas)"]
            
            
        num_archivos_para_cluster = math.ceil(num_incidencias_cluster / INCIDENCIAS_POR_ARCHIVO)

        print(f"  - Cluster '{cluster}': {num_incidencias_cluster} incidencias -> Se generarán {num_archivos_para_cluster} archivo(s).")


        for i in range(num_archivos_para_cluster):
            inicio = i * INCIDENCIAS_POR_ARCHIVO
            fin = inicio + INCIDENCIAS_POR_ARCHIVO
            df_chunk = df_filtrado.iloc[inicio:fin]


            contenido_md = f"# Base de Conocimiento para: {cluster}"
            if num_archivos_para_cluster > 1:
                contenido_md += f" (Parte {i+1}/{num_archivos_para_cluster})"
            contenido_md += "\n\n"
            
            contenido_md += f"Este documento resume los problemas y soluciones comunes para el cluster **{cluster}**.\n\n"
            contenido_md += f"## Temas Clave (Lemas Frecuentes)\n"
            contenido_md += f"`{', '.join(lemas_clave)}`\n\n"
            contenido_md += "---\n\n"
            contenido_md += f"## Problemas y Soluciones ({len(df_chunk)} de {num_incidencias_cluster} en este archivo)\n\n"
            
            for index, row in df_chunk.iterrows():
                contenido_md += f"### Incidencia ID: {row['idincidencia']}\n\n"
                contenido_md += f"**Problema Reportado:**\n"
                contenido_md += f"> {row['problema']}\n\n"
                
                solucion = row['solucion'] if pd.notna(row['solucion']) else "No se especificó una solución."
                contenido_md += f"**Solución Aplicada:**\n"
                contenido_md += f"> {solucion}\n\n"
                contenido_md += "---\n"


            nombre_base = f"{cluster}".replace('/', '_').replace('\\', '_') # Sanear nombre de archivo
            nombre_archivo = f"{nombre_base}_parte_{i+1}.md" if num_archivos_para_cluster > 1 else f"{nombre_base}.md"
            ruta_archivo = os.path.join(path, nombre_archivo)
            
            with open(ruta_archivo, 'w', encoding='utf-8') as f:
                f.write(contenido_md)
            
            total_archivos_creados += 1

    print(f"\n✅ Proceso completado. Se crearon {total_archivos_creados} archivos en total en la carpeta '{path}'.")
