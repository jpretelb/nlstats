import os
import json
import pandas as pd
import spacy
import re
import time
from sklearn.feature_extraction.text import CountVectorizer
from typing import Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE # Para reducción de dimensionalidad
import numpy as np # Para manejo de arrays
from tqdm import tqdm

ARCHIVO_INCIDENCIAS = 'incidencias_consolidadas.csv'
ARCHIVO_SALIDA_LEMAS = 'incidencias_con_lemas.csv'
ARCHIVO_LEMATIZADO = 'incidencias_con_lemas.csv'

def procesar_incidencias_a_csv(carpeta_json, archivo_salida_csv):
    tqdm.pandas()

    """
    Recorre una carpeta de archivos JSON de incidencias, los procesa
    y los consolida en un único archivo CSV aplanado.
    """
    lista_de_incidencias = []
    
    print(f"Buscando archivos JSON en la carpeta: '{carpeta_json}'...")

    for nombre_archivo in os.listdir(carpeta_json):
        if nombre_archivo.endswith('.json'):
            ruta_completa = os.path.join(carpeta_json, nombre_archivo)
            
            try:
                with open(ruta_completa, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    entidades_aplanadas = {
                        'entidad_modulo': None,
                        'entidad_proceso': None,
                        'entidad_documento': None
                    }
                    
                    for entidad in data.get('entidades', []):
                        tipo = entidad.get('tipo')
                        nombre = entidad.get('nombre')
                        
                        if tipo == 'Módulo':
                            entidades_aplanadas['entidad_modulo'] = nombre
                        elif tipo == 'Proceso':
                            entidades_aplanadas['entidad_proceso'] = nombre
                        elif tipo == 'Documento':
                            entidades_aplanadas['entidad_documento'] = nombre
                    
                    fila = {
                        'idincidencia': data.get('idincidencia'),
                        'fechacreacion': data.get('fechacreacion'),
                        'empresa': data.get('empresa'),
                        'modulo': data.get('modulo'),
                        'categoria': data.get('categoria'),
                        'urgencia_estimada': data.get('urgencia_estimada'),
                        'problema': data.get('problema'),
                        'solucion': data.get('solucion'),
                        **entidades_aplanadas  
                    }
                    
                    lista_de_incidencias.append(fila)

            except Exception as e:
                print(f"Error procesando el archivo {nombre_archivo}: {e}")

    if not lista_de_incidencias:
        print("No se encontraron archivos JSON o no se pudieron procesar.")
        return

    df = pd.DataFrame(lista_de_incidencias)
    
    columnas_ordenadas = [
        'idincidencia', 'fechacreacion', 'empresa', 'modulo', 'categoria', 
        'urgencia_estimada', 'problema', 'solucion', 'entidad_modulo', 
        'entidad_proceso', 'entidad_documento'
    ]
    df = df[columnas_ordenadas]

    df.to_csv(archivo_salida_csv, index=False, encoding='utf-8-sig')
    
    print("--------------------------------------------------")
    print(f"¡Proceso completado exitosamente!")
    print(f"Se ha creado el archivo '{archivo_salida_csv}' con {len(df)} filas.")
    print("--------------------------------------------------")


def iniciar():
    start_time = time.time()
    print("\n--- Procesando Incidencias ---")
    print("Cargando modelo de spaCy 'es_core_news_lg' (puede tardar)...")
    nlp = spacy.load("es_core_news_lg")
    print("Modelo cargado.")

    print("--- Cargando listas de palabras ---")
    BLACKLIST_LEMAS = set()
    BLACKLIST_LEMAS = cargar_lista_desde_csv('./datasets/blacklist.csv', 'lema_a_ignorar', BLACKLIST_LEMAS)
    BLACKLIST_LEMAS = cargar_lista_desde_csv('./datasets/apellidos.csv', 'apellido', BLACKLIST_LEMAS)
    BLACKLIST_LEMAS = cargar_lista_desde_csv('./datasets/hombres.csv', 'nombre', BLACKLIST_LEMAS)
    BLACKLIST_LEMAS = cargar_lista_desde_csv('./datasets/mujeres.csv', 'nombre', BLACKLIST_LEMAS)
    print(f"Total de términos en la blacklist final: {len(BLACKLIST_LEMAS)}")

    PROTECTED_WORDS = set()
    PROTECTED_WORDS = cargar_lista_desde_csv('./datasets/protected_words.csv', 'palabra_protegida', PROTECTED_WORDS)


    print(f"Cargando incidencias desde '{ARCHIVO_INCIDENCIAS}'...")
    df_incidencias = pd.read_csv(ARCHIVO_INCIDENCIAS)
    print(f"{len(df_incidencias)} incidencias cargadas.")
    """
    print("Procesando y lematizando problemas...")
    df_incidencias['lemas_problema'] = df_incidencias['problema'].progress_apply(
        lambda texto: lematizar_y_limpiar(texto, nlp, BLACKLIST_LEMAS, PROTECTED_WORDS)
    )

    df_incidencias.to_csv(ARCHIVO_LEMATIZADO, index=False, encoding='utf-8-sig')
    
    
    print("Procesando y lematizando respuestas...")
    df_incidencias['lemas_solucion'] = df_incidencias['problema'].progress_apply(
        lambda texto: lematizar_y_limpiar(texto, nlp, BLACKLIST_LEMAS, PROTECTED_WORDS)
    )
    """
    df_incidencias_lemmas = pd.read_csv(ARCHIVO_LEMATIZADO)
    print(df_incidencias_lemmas)


    """
    ngrams_por_modulo_3 = get_ngrams_from_dataframe(
        df=df_incidencias,
        n_gram_min=3,
        n_gram_max=3, 
        top_k=20
    )

    ngrams_por_modulo_2 = get_ngrams_from_dataframe(
        df=df_incidencias,
        n_gram_min=2,
        n_gram_max=2, 
        top_k=20
    )

    print(agrupar_bigramas_sin_orden(ngrams_por_modulo_3.get('Recursos Humanos')))
    print(agrupar_bigramas_sin_orden(ngrams_por_modulo_2.get('Recursos Humanos')))
    """
    
    modulos = df_incidencias_lemmas['modulo'].unique()

    MIN_INCIDENCIAS_POR_MODULO = 10
    NUMERO_MAX_CLUSTERS = 15
    elbow_data = {} 
    all_clusters_dfs = []

    #for modulo in modulos:

    for modulo in tqdm(modulos, desc="Vectorizando y Clusterizando por Módulo"):
        #if modulo != "RECURSOS HUMANOS":
        #    continue
        print(f"\n--- Procesando Módulo: {modulo} ---")
        df_modulo = df_incidencias_lemmas[df_incidencias_lemmas['modulo'] == modulo].copy()

        if len(df_modulo) < MIN_INCIDENCIAS_POR_MODULO:
            print(f"⚠️  Se omitió el módulo por tener menos de {MIN_INCIDENCIAS_POR_MODULO} incidencias.")
            continue
        
    
        vectorizer = TfidfVectorizer(
            max_df=0.8, 
            min_df=5, 
            max_features=1000,
            ngram_range=(1, 2)
        )
        #vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=1000)

        print(df_modulo['lemas_problema'].fillna(""))
        X = vectorizer.fit_transform(df_modulo['lemas_problema'].fillna(""))
        
        max_k_para_modulo = min(NUMERO_MAX_CLUSTERS, len(df_modulo) - 1)
        rango_k = list(range(2, max_k_para_modulo + 1))
        
        inertia = []
        if len(rango_k) > 1:
            for k in rango_k:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
                inertia.append(kmeans.inertia_)
        
        if not inertia:
            print(" No se pudo calcular el codo. Se omitió el módulo.")
            continue

        """DIBUJAR CODO"""
        print(f"Generando visualización del codo para {modulo}...")

        # 1. Crear el gráfico
        plt.figure(figsize=(8, 6))
        plt.plot(rango_k, inertia, marker='o', linestyle='-', color='b')
        plt.title(f'Método del Codo para el Módulo: {modulo}')
        plt.xlabel('Número de Clusters (k)')
        plt.ylabel('Inercia (SSE)')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Opcional: Marcar el codo detectado por KneeLocator si ya lo has calculado
        kn = KneeLocator(rango_k, inertia, curve='convex', direction='decreasing')
        if kn.elbow:
            plt.axvline(kn.elbow, color='r', linestyle='--', label=f'Codo detectado: {kn.elbow}')
            plt.legend()
        
        plt.tight_layout()
        
        # 2. Guardar el gráfico
        nombre_archivo_codo = f"codo_modulo_{modulo.replace(' ', '_')}.png"
        plt.savefig(nombre_archivo_codo)
        plt.close() # Cierra la figura para liberar memoria
        print(f"Gráfico del codo guardado: {nombre_archivo_codo}")

        """DIBUJAR CODO"""
        
        elbow_data[modulo] = {'k': rango_k, 'inertia': inertia}
        print("------------------------")
        print(kn)
        #print(elbow_data)

        #kn = KneeLocator(rango_k, inertia, curve='convex', direction='decreasing')
        n_clusters_elegido = kn.elbow if kn.elbow else 3
        print(f"Número óptimo de clusters detectado (codo): {n_clusters_elegido}")
        
        kmeans = KMeans(n_clusters=n_clusters_elegido, random_state=42, n_init=10)
        df_modulo['cluster_label'] = kmeans.fit_predict(X)
        df_modulo['cluster_consolidado'] = df_modulo['modulo'] + '_Cluster_' + df_modulo['cluster_label'].astype(str)
        
        print(f"--- Top 10 términos por Cluster para {modulo} ---")
        order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names_out()
        for i in range(n_clusters_elegido):
            top_terms = [terms[ind] for ind in order_centroids[i, :10]]
            print(f"Cluster {i}: {', '.join(top_terms)}")
        
        
        print(f"Generando visualización de clusters para {modulo}...")
        try:
            # Asegúrate de que X no esté vacío y tenga suficientes muestras
            if X.shape[0] < 2 or n_clusters_elegido < 2:
                print("Insuficientes datos o clusters para t-SNE. Saltando visualización.")
                all_clusters_dfs.append(df_modulo)
                continue

            # Limitamos el número de iteraciones y perplexity para dataset pequeños
            # También random_state para reproducibilidad-
            tsne = TSNE(n_components=2, init='random', random_state=42, n_jobs=-1)
            
            X_reduced = tsne.fit_transform(X)

            # Crear el gráfico
            plt.figure(figsize=(10, 8))
            
            # Obtener una lista de colores para los clusters
            colors = plt.cm.get_cmap('tab10', n_clusters_elegido) 
            
            for i in range(n_clusters_elegido):
                # Seleccionar puntos que pertenecen a este cluster
                points = X_reduced[df_modulo['cluster_label'] == i]
                plt.scatter(points[:, 0], points[:, 1], 
                            s=50, # Tamaño de los puntos
                            c=[colors(i)], # Color del cluster
                            label=f'Cluster {i}', 
                            alpha=0.7) # Transparencia
            
            plt.title(f'Visualización de Clusters para el Módulo: {modulo} (K={n_clusters_elegido})')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Guardar el gráfico
            nombre_archivo_grafico = f"clusters_modulo_{modulo.replace(' ', '_')}.png"
            print(nombre_archivo_grafico)
            plt.savefig(nombre_archivo_grafico)
            plt.close() # Cierra la figura para liberar memoria
            print(f"Gráfico guardado: {nombre_archivo_grafico}")

            

        except Exception as e:
            print(f"❌ Error al generar el gráfico t-SNE para el módulo {modulo}: {e}")
            print("Asegúrate de tener suficientes datos para t-SNE (al menos 2 muestras).")

        all_clusters_dfs.append(df_modulo)
    
    ARCHIVO_SALIDA_CLUSTERS = 'incidencias_con_clusters_por_modulo.csv'
    df_final = pd.concat(all_clusters_dfs)
    df_final = df_final[['idincidencia', 'modulo', 'lemas_problema', 'cluster_consolidado']]
    df_final.to_csv(ARCHIVO_SALIDA_CLUSTERS, index=False, encoding='utf-8-sig')
    print(f"\n-Archivo con clusters por módulo guardado en '{ARCHIVO_SALIDA_CLUSTERS}'.")

    #with open(ARCHIVO_SALIDA_ELBOW, 'w') as f:
    #    json.dump(elbow_data, f)
    #print(f"-Datos del método del codo guardados en '{ARCHIVO_SALIDA_ELBOW}'.")



    #print("Lematización completada.")
#
    #print(f"\n--- Guardando Resultados ---")
    #df_salida = df_incidencias[['idincidencia', 'lemas_procesados']]
    #df_salida.to_csv(ARCHIVO_SALIDA_LEMAS, index=False, encoding='utf-8-sig')
    #print(f"Archivo de mapeo '{ARCHIVO_SALIDA_LEMAS}' creado exitosamente!")
#
    #end_time = time.time()
    #print(f"\nProceso completado en {end_time - start_time:.2f} segundos.")



def agrupar_bigramas_sin_orden(df_bigramas: pd.DataFrame) -> pd.DataFrame:
    """
    Agrupa bigramas que contienen las mismas palabras, independientemente de su orden.
    
    Args:
        df_bigramas (pd.DataFrame): DataFrame con columnas 'ngram' y 'frecuencia'.
        
    Returns:
        pd.DataFrame: Nuevo DataFrame con los bigramas agrupados y sus frecuencias sumadas.
    """
    
    # 1. Crear una clave de ordenamiento (Normalización)
    # Ejemplo: 'calcular planilla' -> ['calcular', 'planilla'] -> sorted -> ['calcular', 'planilla'] -> 'calcular planilla'
    # Ejemplo: 'planilla calcular' -> ['planilla', 'calcular'] -> sorted -> ['calcular', 'planilla'] -> 'calcular planilla'
    
    # Esta función toma el bigrama, lo separa por espacios, ordena las palabras 
    # y las vuelve a unir, creando una clave única.
    df_bigramas['clave_normalizada'] = df_bigramas['ngram'].apply(
        lambda x: ' '.join(sorted(x.split()))
    )
    
    # 2. Agrupar y Sumar Frecuencias
    # Agrupamos por la clave normalizada y sumamos las frecuencias.
    df_agrupado = df_bigramas.groupby('clave_normalizada')['frecuencia'].sum().reset_index()
    
    # 3. Renombrar las columnas para el resultado final
    df_agrupado.columns = ['bigrama_agrupado', 'frecuencia_total']
    
    # 4. Ordenar y devolver
    df_agrupado = df_agrupado.sort_values(by='frecuencia_total', ascending=False)
    
    return df_agrupado

    
def get_ngrams_from_dataframe(
    df: pd.DataFrame, 
    text_column: str = 'lemas_problema',
    group_column: str = 'modulo',
    n_gram_min: int = 1, 
    n_gram_max: int = 2, 
    top_k: int = 10
) -> Dict[str, pd.DataFrame]:
    """
    Calcula los N-gramas más frecuentes para cada grupo (módulo) en un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de Pandas con todas las columnas.
        text_column (str): Nombre de la columna con el texto/lemas procesados.
        group_column (str): Nombre de la columna para agrupar (ej. 'modulo').
        n_gram_min (int): Tamaño mínimo del N-grama (ej. 1 para unigramas).
        n_gram_max (int): Tamaño máximo del N-grama (ej. 2 para bigramas).
        top_k (int): Número de N-gramas más frecuentes a devolver por grupo.

    Returns:
        Dict[str, pd.DataFrame]: Diccionario con el módulo como clave y un DataFrame 
                                 de los N-gramas y sus frecuencias como valor.
    """
    # 1. Validación y Limpieza
    if group_column not in df.columns or text_column not in df.columns:
        raise KeyError(f"El DataFrame debe contener las columnas '{group_column}' y '{text_column}'.")
        
    df = df.copy()
    df[text_column] = df[text_column].fillna('').astype(str)
    
    # 2. Agregar lemas por 'modulo'
    # Concatenar todos los lemas de cada módulo en un gran texto (corpus)
    df_grouped = df.groupby(group_column)[text_column].apply(lambda x: ' '.join(x)).reset_index()
    df_grouped.columns = [group_column, 'corpus_grupo']

    results = {}

    # 3. Calcular N-gramas para cada módulo/grupo
    #for index, row in df_grouped.iterrows():
    for index, row in tqdm(df_grouped.iterrows(), total=len(df_grouped)):
        group_name = row[group_column]
        corpus = row['corpus_grupo']

        if not corpus.strip():
            print(f"⚠️ Corpus vacío para el módulo {group_name}. Omitiendo.")
            results[group_name] = pd.DataFrame(columns=['ngram', 'frecuencia'])
            continue # <--- Esto debería prevenir el error, pero revisa tu lógica de limpieza.

        # Si el corpus está vacío, se salta o se registra un resultado vacío
        if not corpus.strip():
            results[group_name] = pd.DataFrame(columns=['ngram', 'frecuencia'])
            continue

        # Inicializar CountVectorizer
        # ngram_range=(n_gram_min, n_gram_max) define el rango de N-gramas
        vectorizer = CountVectorizer(
            ngram_range=(n_gram_min, n_gram_max),
            token_pattern=r'\b\w+\b' # Patrón para asegurar que solo se tomen palabras
        )
        
        # Generar matriz de conteos
        X = vectorizer.fit_transform([corpus])
        
        # Obtener los N-gramas (features) y sus frecuencias
        feature_names = vectorizer.get_feature_names_out()
        
        counts = X.toarray().sum(axis=0)
        
        # Crear DataFrame de resultados para el módulo
        df_ngrams = pd.DataFrame({
            'ngram': feature_names,
            'frecuencia': counts
        })
        
        # Ordenar por frecuencia (descendente) y seleccionar los top K
        df_ngrams = df_ngrams.sort_values(by='frecuencia', ascending=False).head(top_k).reset_index(drop=True)
        
        results[group_name] = df_ngrams

    return results


def lematizar_y_limpiar(texto, nlp_model, blacklist, protected_list):
    """
    Procesa texto con limpieza Regex y spaCy, protegiendo la whitelist y eliminando duplicados consecutivos.
    """
    if not isinstance(texto, str):
        return ""
        
    texto = re.sub(r'\d+/\d+(/\d+)?', ' ', texto)
    texto = re.sub(r'\d+-\d+', ' ', texto)
    texto = re.sub(r'[a-zA-Z]+-\d+-\d+', ' ', texto)
    texto = re.sub(r'\S+@\S+', ' ', texto)
    texto = re.sub(r'https?://\S+', ' ', texto)
    texto = re.sub(r'http?://\S+', ' ', texto)
    texto = re.sub(r'[0-9%]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    texto = texto.replace('-', ' ')

    doc = nlp_model(texto)
    final_tokens = []
    for token in doc:
        """
        print("----")
        print(token.text.lower())
        print(token.lemma_.lower())
        print(token.pos_)
        print(token.is_stop)
        print(token.is_punct)
        print(token.is_digit)
        print(token.like_num)
        print("----")
        """
        
        if (token.is_stop or token.is_punct or token.is_digit or token.like_num):
            continue

        token_text_lower = token.text.lower()
        lemma_text_lower = token.lemma_.lower()

        lemma_parts = lemma_text_lower.split()
        
        # 2. Filtrar las partes que están en la blacklist.
        partes_filtradas = [part for part in lemma_parts if part not in blacklist]
        
        # 3. Unir las partes restantes para formar el lema procesado.
        lemma_procesado = " ".join(partes_filtradas)
        
        # 4. Si después de filtrar no queda nada (ej. todas las partes estaban en la blacklist),
        #    descartamos el token por completo y pasamos al siguiente.
        if not lemma_procesado:
            continue
        
        if token_text_lower in blacklist or lemma_procesado in blacklist:
            continue

        # Lógica para decidir si proteger o usar el lema procesado
        token_a_agregar = None
        if token_text_lower in protected_list:
            token_a_agregar = token_text_lower
        else:
            token_a_agregar = lemma_procesado
        
        if token_a_agregar and (not final_tokens or token_a_agregar != final_tokens[-1]):
            final_tokens.append(token_a_agregar)
            
    return " ".join(final_tokens)

def cargar_lista_desde_csv(filepath, column_name, initial_set):
    """Función auxiliar para cargar palabras desde un CSV a un set."""
    try:
        df = pd.read_csv(filepath)
        initial_set.update(set(df[column_name].str.lower()))
        print(f"{len(df)} términos cargados desde '{filepath}'.")
    except FileNotFoundError:
        print(f"Advertencia: No se encontró '{filepath}'.")
    except KeyError:
        print(f"Error: La columna '{column_name}' no se encontró en '{filepath}'.")
    return initial_set

def obtener_n_gramas_frecuentes(corpus, n_grama_range, top_k=20):
    """
    Encuentra los n-gramas más frecuentes en un corpus ya preprocesado.
    """
    try:
        vec = CountVectorizer(ngram_range=n_grama_range).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0) 
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
        return words_freq[:top_k]
    except ValueError:
        print(f"No se pudieron generar {n_grama_range}-gramas. El vocabulario puede ser muy pequeño.")
        return []
    