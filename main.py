from constants import PATH_CLUSTER, PATH_LEMMA, PATH_INCIDENCIAS, PATH_IMGS_CODO, PATH_IMGS_CLUSTER
from constants import PATH_DATASET, PATH_WHITE_LIST, PATH_BLACK_LIST, PATH_THEME, PATH_KB

from constants import PATH_CHROMA_DB

import argparse
import sys
import os # Para las variables de entorno
import pyfiglet
from config_utils import load_db_config, load_gemini_token
from fetch_data import run_fetch
from gemini import run_normaliza_gemini, rag

from kmeans import clustering_module

#from crear_consolidado import procesar_incidencias_a_csv
from knowledge_base import generar_kb

from lemma import lemmatizar_modulo

from consolidar import procesar_incidencias_a_csv




#ARCHIVO_LEMATIZADO = 'incidencias_con_lemas.csv'
#ARCHIVO_CONSOLIDADO = 'incidencias_consolidadas.csv'
#ARCHIVO_SALIDA_CLUSTERS = 'incidencias_con_clusters_por_modulo.csv'
#ARCHIVO_SALIDA_THEMES  = 'cluster_themes.csv'
#DIRECTORIO_CONOCIMIENTO = 'conocimiento_consolidado'


LOTE_EMBEDDING = 10 

def main():
    """Función principal que configura el parser de argumentos."""

    # 1. Mensaje de Bienvenida
    
    print("=================================================================")
    ascii_art = pyfiglet.figlet_format("Welcome to", font="slant")
    print(ascii_art)
    ascii_art = pyfiglet.figlet_format("NLStats", font="slant")
    print(ascii_art)
    print("=================================================================")
    parser = argparse.ArgumentParser(
        description="NLStats: Herramienta de procesamiento de lenguaje natural.",
        formatter_class=argparse.RawTextHelpFormatter # Mantiene el formato en la ayuda
    )

    os.makedirs(PATH_CLUSTER, exist_ok=True)
    os.makedirs(PATH_LEMMA, exist_ok=True)
    os.makedirs(PATH_INCIDENCIAS, exist_ok=True)
    os.makedirs(PATH_IMGS_CODO, exist_ok=True)
    os.makedirs(PATH_IMGS_CLUSTER, exist_ok=True)
    os.makedirs(PATH_DATASET, exist_ok=True)
    os.makedirs(PATH_WHITE_LIST, exist_ok=True)
    os.makedirs(PATH_BLACK_LIST, exist_ok=True)
    os.makedirs(PATH_THEME, exist_ok=True)
    os.makedirs(PATH_KB, exist_ok=True)
    os.makedirs(PATH_CHROMA_DB, exist_ok=True)

    


    # Configuración de Subcomandos
    subparsers = parser.add_subparsers(
        title='Comandos disponibles',
        dest='command' # Variable que contendrá el nombre del subcomando ejecutado
    )
    subparsers.required = True

    # ----------------------------------------------------
    # Subcomando 1: 'fetch' (Bajar datos de MSSQL a Postgres)
    # ----------------------------------------------------
    parser_fetch = subparsers.add_parser('fetch', help='Bajar datos de MSSQL a Postgres local.')
    parser_fetch.add_argument(
        '-m', '--mssql-pass',
        required=True,
        help='Clave de acceso para la BD origen MSSQL.'
    )
    parser_fetch.add_argument(
        '-p', '--postgres-pass',
        required=True,
        help='Clave de acceso para la BD local Postgres.'
    )
    parser_fetch.add_argument(
        '-sd', '--start-date',
        required=True,
        help='Start Desde en formato YYYYMMDD.'
    )
    parser_fetch.add_argument(
        '-ed', '--end-date',
        required=True,
        help='Fecha Hasta en formato YYYYMMDD.'
    )
    parser_fetch.set_defaults(func=fetch_data)

    # ----------------------------------------------------
    # Subcomando 2: 'normalize' (Normalización con Gemini)
    # ----------------------------------------------------
    parser_normalize = subparsers.add_parser('ia_normalize', help='Normalización de datos con IA.')
    parser_normalize.add_argument(
        '-p', '--postgres-pass',
        required=True,
        help='Clave de acceso para la BD local Postgres.'
    )
    parser_normalize.add_argument(
        '-m', '--month',
        required=True,
        help='Mes en formato YYYYMM.'
    )
    parser_normalize.set_defaults(func=normalize_data)


    parser_consolidate = subparsers.add_parser('consolidate', help='Consolidar la información de las incidencias.')
    parser_consolidate.set_defaults(func=consolidate_data)


    # ----------------------------------------------------
    # Subcomando 3: 'analyze' (Análisis de frecuencias)
    # ----------------------------------------------------
    parser_analyze = subparsers.add_parser('proc', help='Análisis de frecuencias (trabaja con archivos JSON).')

    parser_analyze.add_argument(
        '--lemma',
        action='store_true',
        help='Paso 2: Aplica lematización a los datos consolidados.'
    )

    parser_analyze.add_argument(
        '--cluster',
        action='store_true',
        help='Paso 3: Ejecuta el algoritmo de clustering sobre los datos lematizados.'
    )

    parser_analyze.add_argument(
        '--kb',
        action='store_true',
        help='Paso 4: Genera archivos para la base de conocimiento.'
    )

    parser_analyze.add_argument(
        '--rag',
        action='store_true',
        help='Paso 5: Creación de Retrieval-Augmented Generation..'
    )

    parser_analyze.add_argument(
        '-m', '--module',
        required=True,
        help='Modulo.'
    )

    parser_analyze.add_argument(
        '--maxdf',
        required=False,
        help='Frecuencia máxima del documento: Valor entre 0.0 y 1.0. Se ignoran las palabras que aparecen en más del porcentaje especificado de documentos.'
    )

    parser_analyze.add_argument(
        '--mindf',
        required=False,
        help='Frecuencia mínima del documento: Valor 0.0 y 1.0. Se ignoran las palabras que aparecen en menos del porcentaje especificado de documentos.'
    )

    




    # Las opciones para este comando son opcionales ya que trabaja con archivos preexistentes.
    parser_analyze.set_defaults(func=analyze_data)

    # ----------------------------------------------------
    # Procesar y Ejecutar
    # ----------------------------------------------------
    if len(sys.argv) == 1:
        # Si no se pasan argumentos, muestra la ayuda general
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Ejecutar la función asociada al subcomando
    args.func(args)


# ----------------------------------------------------
# Funciones de Comando
# ----------------------------------------------------

def fetch_data(args):
    """Lógica para el comando 'fetch'."""
    print("\n--- Recuperando informacion ---")
    
    postgres_conf = load_db_config('postgres')

    mssql_conf = load_db_config('mssql')
    
    mssql_pass = args.mssql_pass
    postgres_pass = args.postgres_pass
    start_date = args.start_date
    end_date = args.end_date

    #Cargar Token de Gemini (SEGURIDAD)
    #gemini_token = load_gemini_token()

    try:
        run_fetch(mssql_conf, postgres_conf, mssql_pass, postgres_pass, start_date, end_date)
        print("\nMigración completada exitosamente.")
    except Exception as e:
        print(f"\nError fatal durante la migración: {e}")
    # Aquí iría la lógica real de conexión/migración.

def normalize_data(args):
    """Lógica para el comando 'normalize'."""
    print("\n--- Normalizando data con IA ---")
    
    postgres_pass = args.postgres_pass
    month = args.month
    postgres_conf = load_db_config('postgres')

    try:
        run_normaliza_gemini(postgres_conf, postgres_pass, month)
        print("\nNormalizacion completada exitosamente.")
    except Exception as e:
        print(f"\nError fatal durante la normalización: {e}")
        print(e)

def consolidate_data(args):
    print(f"Consolidating data....")
    procesar_incidencias_a_csv()

def analyze_data(args):
    """Lógica para el comando 'analyze'."""
    print("\n--- EJECUTANDO COMANDO: ANALYZE ---")
    print("Analizando frecuencias a partir de archivos JSON existentes...")

    lemmatize = False
    clustering = False
    kb = False
    israg = False

    if args.lemma:
        print("\nLematizando incidencias")
        try:
            lemmatizar_modulo(modulo=args.module)
        except Exception as e:
            print(f"Error en el paso de Lemmatizacion: {e}")
            sys.exit(1)

    if args.cluster:
        print("\nClustering...")
        try:
            clustering_module(args.module, float(args.maxdf), int(args.mindf))

        except Exception as e:
            print(f"Error Clustering: {e}")
            sys.exit(1)
    
    
    if args.kb:
        print("\nKB...")
        try:
            generar_kb(args.module, 'lemas_problema')
        except Exception as e:
            print(f"Error Clustering: {e}")
            sys.exit(1)


    
    if args.rag:
        print("\nRAG...")
        try:
            
            rag(args.module)
        except Exception as e:
            print(f"Error Clustering: {e}")
            sys.exit(1)
    

    # Aquí iría la lógica real de análisis de frecuencias.

if __name__ == '__main__':
    main()