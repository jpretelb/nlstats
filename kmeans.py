import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from kneed import KneeLocator
import numpy as np

from gemini import return_etiquetas_from_cluster

def clustering_algorithm(csv_file, csv_out, csv_out_themes, module=None):
    """Ejecuta el algoritmo de clustering K-Means en el archivo CSV proporcionado.
    Args:
        csv_file (str): Ruta al archivo CSV con los datos a clusterizar.
    """

    df_incidencias_lemmas = pd.read_csv(csv_file)
    
    modulos = df_incidencias_lemmas['modulo'].unique()
    MIN_INCIDENCIAS_POR_MODULO = 10
    NUMERO_MAX_CLUSTERS = 60
    elbow_data = {} 
    all_clusters_dfs = []
    cluster_themes = []

    for modulo in tqdm(modulos, desc="Vectorizando y Clusterizando por Módulo"):
        img_name = modulo.replace(' ', '_').replace('-', '_') + ".png"
        if module and modulo != module:
            continue
        #if modulo != "RECURSOS HUMANOS":
        #    continue
        try:
            
            print(f"\n--- Procesando Módulo: {modulo} ---")
            df_modulo = df_incidencias_lemmas[df_incidencias_lemmas['modulo'] == modulo].copy()

            if len(df_modulo) < MIN_INCIDENCIAS_POR_MODULO:
                print(f"⚠️  Se omitió el módulo por tener menos de {MIN_INCIDENCIAS_POR_MODULO} incidencias.")
                continue
            
        
            vectorizer = TfidfVectorizer(
                max_df=0.3, 
                min_df=3, 
                max_features=5000,
                ngram_range=(1, 3)
            )
            #vectorizer = TfidfVectorizer(max_df=0.8, min_df=5, max_features=1000)

            
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
            nombre_archivo_codo = ombre_archivo_grafico = "./imgs/codo/" + img_name #f"codo_modulo_{modulo.replace(' ', '_')}.png"
            plt.savefig(nombre_archivo_codo)
            plt.close() # Cierra la figura para liberar memoria
            print(f"Gráfico del codo guardado: {nombre_archivo_codo}")

            """DIBUJAR CODO"""
            
            elbow_data[modulo] = {'k': rango_k, 'inertia': inertia}
            #print("------------------------")
            #print(kn)
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
                cluster_themes.append({
                    'modulo': modulo,
                    'cluster': i,
                    'top_terms': top_terms
                })
            
            
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
                nombre_archivo_grafico = "./imgs/cluster/" + img_name #f"clusters_modulo_{modulo.replace(' ', '_')}.png"
                print(nombre_archivo_grafico)
                plt.savefig(nombre_archivo_grafico)
                plt.close() # Cierra la figura para liberar memoria
                print(f"Gráfico guardado: {nombre_archivo_grafico}")

                

            except Exception as e:
                print(f"Error al generar el gráfico t-SNE para el módulo {modulo}: {e}")
                print("Asegúrate de tener suficientes datos para t-SNE (al menos 2 muestras).")

            all_clusters_dfs.append(df_modulo)
        except Exception as e:
            print(f"Error al clusterizar el módulo {modulo}: {e}")

    
    df_final = pd.concat(all_clusters_dfs)
    df_final = df_final[['idincidencia', 'modulo', 'lemas_problema', 'cluster_consolidado']]
    df_final.to_csv(csv_out, index=False, encoding='utf-8-sig')
    print(f"\n-Archivo con clusters por módulo guardado en '{csv_out}'.")


    

    df_final_themes = pd.DataFrame(cluster_themes)
    df_final_themes = df_final_themes[['modulo','cluster','top_terms']]
    #print(cluster_themes)
    #print(df_final_themes)
    df_final_themes.to_csv(csv_out_themes, index=False, encoding='utf-8-sig')

    
def clustering_module(modulo, p_max_df, p_min_df):
    csv_name = "./lemma/" + modulo.replace(' ', '_').replace('-', '_') + ".csv"
    csv_name_cluster = "./cluster/" + modulo.replace(' ', '_').replace('-', '_') + ".csv"
    csv_name_themes = "./themes/" + modulo.replace(' ', '_').replace('-', '_') + ".csv"

    """Ejecuta el algoritmo de clustering K-Means en el archivo CSV proporcionado.
    Args:
        csv_file (str): Ruta al archivo CSV con los datos a clusterizar.
    """
    print(f"\n--- Procesando Módulo: {modulo} con parametros max_df={p_max_df}, min_df={p_min_df} ---")
    
    img_name = modulo.replace(' ', '_').replace('-', '_') + ".png"

    print("Loading data...")
    df_incidencias_lemmas = pd.read_csv(csv_name)

    print("Filtering data...")
    df_modulo = df_incidencias_lemmas[df_incidencias_lemmas['modulo'] == modulo].copy()

    elbow_data = {} 
    all_clusters_dfs = []
    cluster_themes = []

    MIN_INCIDENCIAS_POR_MODULO = 10
    NUMERO_MAX_CLUSTERS = 60
    
    vectorizer = TfidfVectorizer(
        max_df=p_max_df, 
        min_df=p_min_df, 
        max_features=5000,
        ngram_range=(1, 3)
    )

    X = vectorizer.fit_transform(df_modulo['lemas_problema'].fillna(""))
            
    max_k_para_modulo = min(NUMERO_MAX_CLUSTERS, len(df_modulo) - 1)
    rango_k = list(range(2, max_k_para_modulo + 1))
    
    if len(df_modulo) < MIN_INCIDENCIAS_POR_MODULO:
        print(f"⚠️  Se omitió el módulo por tener menos de {MIN_INCIDENCIAS_POR_MODULO} incidencias.")
        return
    
    inertia = []
    if len(rango_k) > 1:
        for k in rango_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
            inertia.append(kmeans.inertia_)


    if not inertia:
        print(" No se pudo calcular el codo. Se omitió el módulo.")
        return
    
    
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
    nombre_archivo_codo = ombre_archivo_grafico = "./imgs/codo/" + img_name #f"codo_modulo_{modulo.replace(' ', '_')}.png"
    plt.savefig(nombre_archivo_codo)
    plt.close() # Cierra la figura para liberar memoria
    print(f"Gráfico del codo guardado: {nombre_archivo_codo}")
    
    """DIBUJAR CODO"""

    elbow_data[modulo] = {'k': rango_k, 'inertia': inertia}
    #print("------------------------")
    #print(kn)
    #print(elbow_data
    #kn = KneeLocator(rango_k, inertia, curve='convex', direction='decreasing')
    n_clusters_elegido = kn.elbow if kn.elbow else 3
    print(f"Número óptimo de clusters detectado (codo): {n_clusters_elegido}")
    
    kmeans = KMeans(n_clusters=n_clusters_elegido, random_state=42, n_init=10)
    df_modulo['cluster_label'] = kmeans.fit_predict(X)
    df_modulo['cluster_consolidado'] = df_modulo['cluster_label'].astype(str)
    
    centroids = kmeans.cluster_centers_
    n_clusters_elegido = centroids.shape[0]
    n_features = centroids.shape[1]
    discriminative_weights = np.zeros_like(centroids)
    for i in range(n_clusters_elegido):
        # Crear una máscara booleana para seleccionar todos los clusters excepto 'i'
        mask = np.ones(n_clusters_elegido, dtype=bool)
        mask[i] = False
        
        # a. Obtener los centroides de los 'otros' clusters
        other_centroids = centroids[mask, :]
        
        # b. Calcular el peso promedio de cada término en los 'otros' clusters
        # Promediamos a lo largo del eje 0 (los clusters)
        mean_other_weights = other_centroids.mean(axis=0)
        
        # c. Calcular la diferencia: Peso_Cluster_i - Peso_Promedio_Otros
        # Esto destaca los términos que son únicos para el cluster 'i'
        discriminative_weights[i] = centroids[i] - mean_other_weights

    # 3. Ordenar los nuevos pesos para obtener los índices discriminatorios
    # ArgSort ordena y [:, ::-1] invierte para obtener el orden descendente
    discriminative_order = discriminative_weights.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names_out()
    for i in range(n_clusters_elegido):
        # ¡USANDO LA NUEVA ORDENACIÓN DISCRIMINATORIA!
        top_terms = [terms[ind] for ind in discriminative_order[i, :25]]
        
        #print(f"Cluster {i}: {', '.join(top_terms)}")
        cluster_themes.append({
            'modulo': modulo,
            'cluster': i,
            'top_terms': top_terms
        })
        

    #print(f"--- Top 10 términos por Cluster para {modulo} ---")
    #order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    #terms = vectorizer.get_feature_names_out()
    #for i in range(n_clusters_elegido):
    #    top_terms = [terms[ind] for ind in order_centroids[i, :15]]
    #    #print(f"Cluster {i}: {', '.join(top_terms)}")
    #    cluster_themes.append({
    #        'modulo': modulo,
    #        'cluster': i,
    #        'top_terms': top_terms
    #    })

    print(f"Generando visualización de clusters para {modulo}...")
    try:
        # Asegúrate de que X no esté vacío y tenga suficientes muestras
        if X.shape[0] < 2 or n_clusters_elegido < 2:
            print("Insuficientes datos o clusters para t-SNE. Saltando visualización.")
            all_clusters_dfs.append(df_modulo)
            return
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
        nombre_archivo_grafico = "./imgs/cluster/" + img_name #f"clusters_modulo_{modulo.replace(' ', '_')}.png"
        print(nombre_archivo_grafico)
        plt.savefig(nombre_archivo_grafico)
        plt.close() # Cierra la figura para liberar memoria
        print(f"Gráfico guardado: {nombre_archivo_grafico}")
    except Exception as e:
        print(f"Error al generar el gráfico t-SNE para el módulo {modulo}: {e}")
        print("Asegúrate de tener suficientes datos para t-SNE (al menos 2 muestras).")
    
    all_clusters_dfs.append(df_modulo)

    df_final = pd.concat(all_clusters_dfs)
    df_final = df_final[['idincidencia', 'modulo', 'lemas_problema', 'cluster_consolidado']]
    df_final.to_csv(csv_name_cluster, index=False, encoding='utf-8-sig')
    print(f"\n-Archivo con clusters por módulo guardado en '{csv_name_cluster}'.")


    texto_completo = ""
    for cluster in cluster_themes:
        modulo = cluster["modulo"]
        cluster_id = cluster["cluster"]
        top_terms = cluster["top_terms"]
        texto_completo += f"\n\n### Cluster ID: {cluster_id}\n  - Módulo: {modulo}\n  - Términos más relevantes: {', '.join(top_terms)}"
    
    #print(texto_completo)

    ia_themes = return_etiquetas_from_cluster(texto_completo)

    tema_map = {item['cluster_id']: item['tema'] for item in ia_themes}

    for item in cluster_themes:
        cluster_id = item['cluster']
        
        if cluster_id in tema_map:
            item['ia_theme'] = tema_map[cluster_id]
        else:
            item['ia_theme'] = "Tema no encontrado"
            

    df_final_themes = pd.DataFrame(cluster_themes)
    df_final_themes = df_final_themes[['modulo','cluster','top_terms', 'ia_theme']]
    #print(cluster_themes)
    #print(df_final_themes)
    df_final_themes.to_csv(csv_name_themes, index=False, encoding='utf-8-sig')
    
    

    print(ia_themes)

    