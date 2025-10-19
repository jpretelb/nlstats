import spacy
import re
import time
import pandas as pd
from tqdm import tqdm

def lemmatizar_modulo(csv_input, modulo, idincidencia=None):
    csv_name = "./lemma/" + modulo.replace(' ', '_').replace('-', '_') + ".csv"

    tqdm.pandas()
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

    print(f"Cargando incidencias desde '{csv_input}'...")
    df_incidencias = pd.read_csv(csv_input)

    df_modulo = df_incidencias[df_incidencias['modulo'] == modulo].copy()

    #df_modulo = df_incidencias[df_incidencias["idincidencia"] == 230032].copy()

    print(f"{len(df_modulo)} incidencias cargadas.")

    print("Procesando y lematizando problemas...")
    df_modulo['lemas_problema'] = df_modulo['problema'].progress_apply(
        lambda texto: lematizar_y_limpiar(texto, nlp, BLACKLIST_LEMAS, PROTECTED_WORDS)
    )

    
    print("Procesando y lematizando respuestas...")
    df_modulo['lemas_solucion'] = df_modulo['solucion'].progress_apply(
        lambda texto: lematizar_y_limpiar(texto, nlp, BLACKLIST_LEMAS, PROTECTED_WORDS)
    )


    df_modulo.to_csv(csv_name, index=False, encoding='utf-8-sig')


    
def lemmatizar(csv_input, csv_output):
    tqdm.pandas()
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

    print(f"Cargando incidencias desde '{csv_input}'...")
    df_incidencias = pd.read_csv(csv_input)
    print(f"{len(df_incidencias)} incidencias cargadas.")

    print("Procesando y lematizando problemas...")
    df_incidencias['lemas_problema'] = df_incidencias['problema'].progress_apply(
        lambda texto: lematizar_y_limpiar(texto, nlp, BLACKLIST_LEMAS, PROTECTED_WORDS)
    )

    
    print("Procesando y lematizando respuestas...")
    df_incidencias['lemas_solucion'] = df_incidencias['problema'].progress_apply(
        lambda texto: lematizar_y_limpiar(texto, nlp, BLACKLIST_LEMAS, PROTECTED_WORDS)
    )


    df_incidencias.to_csv(csv_output, index=False, encoding='utf-8-sig')



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

    #texto = re .sub(r'R5ta', 'renta de quinta categoria', texto)

    
    #texto = re.sub(r'[0-9%]', '', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()

    #texto = texto.replace('r5ta', ' renta de quinta categoria ')
    #texto = texto.replace('R5TA', ' renta de quinta categoria ')


    texto = texto.replace('-', ' ')

    doc = nlp_model(texto)
    final_tokens = []
    for token in doc:
        
        #print(f"T: {token.text}")
        #print(f"L:: {token.lemma_}")


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
    #print(" ".join(final_tokens))
    return " ".join(final_tokens)