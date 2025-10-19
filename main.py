import argparse
import sys
import os # Para las variables de entorno
import pyfiglet
from config_utils import load_db_config, load_gemini_token
from step0_migration import run_migration
from gemini import run_normaliza_gemini, rag

from kmeans import clustering_module

from crear_consolidado import procesar_incidencias_a_csv, iniciar
from knowledge_base import generar_kb

from lemma import lemmatizar_modulo

ARCHIVO_LEMATIZADO = 'incidencias_con_lemas.csv'
ARCHIVO_CONSOLIDADO = 'incidencias_consolidadas.csv'
ARCHIVO_SALIDA_CLUSTERS = 'incidencias_con_clusters_por_modulo.csv'
DIRECTORIO_SALIDA_MD = 'conocimiento_consolidado'

ARCHIVO_SALIDA_THEMES  = 'cluster_themes.csv'


DIRECTORIO_CONOCIMIENTO = 'conocimiento_consolidado'
DIRECTORIO_CHROMA_DB = 'knowledge_base_db'
NOMBRE_COLECCION = 'problems'
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

    # ----------------------------------------------------
    # Subcomando 3: 'analyze' (Análisis de frecuencias)
    # ----------------------------------------------------
    parser_analyze = subparsers.add_parser('analyze', help='Análisis de frecuencias (trabaja con archivos JSON).')
    parser_analyze.add_argument(
        '--consolidate',
        action='store_true',
        help='Paso 1: Consolida archivos JSON en un único CSV.'
    )

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
        '--only',
        action='store_true',
        help='Unicamente ejecutar el paso solicitado'
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
        run_migration(mssql_conf, postgres_conf, mssql_pass, postgres_pass, start_date, end_date)
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


def analyze_data(args):
    """Lógica para el comando 'analyze'."""
    print("\n--- EJECUTANDO COMANDO: ANALYZE ---")
    print("Analizando frecuencias a partir de archivos JSON existentes...")

    lemmatize = False
    clustering = False
    kb = False
    israg = False

    only = args.only

    if args.consolidate:
        print("\n[Paso 1/3] Consolidando incidencias")
        try:
            
            print(f"Consolidado")
            if not only:
                lemmatize = True
                clustering = True
                kb = True
        except Exception as e:
            print(f"Error en el paso de consolidación: {e}")
            sys.exit(1)
    else:
        print("[Paso 1/3] Consolidando incidencias. Se usará archivos existentes.")

    if args.lemma or lemmatize:
        print("\n[Paso 2/3] Lematizando incidencias")
        try:
            
            print(f"Lematizando incidencias")

            lemmatizar_modulo(csv_input=ARCHIVO_CONSOLIDADO, modulo=args.module)
            if not only:
                clustering = True
                kb = True
        except Exception as e:
            print(f"Error en el paso de Lemmatizacion: {e}")
            sys.exit(1)
    else:
        print("[Paso 2/3] Lematizando incidencias. Se usará archivos existentes.")

    if args.cluster or clustering:
        print("\n[Paso 3/3] Clustering...")
        try:
            
            clustering_module(args.module, float(args.maxdf), int(args.mindf))
            if not only:
                kb = True
        except Exception as e:
            print(f"Error Clustering: {e}")
            sys.exit(1)
    else:
        print("[Paso 2/3] Lematizando incidencias. Se usará archivos existentes.")
    
    if args.kb or kb:
        print("\n[Paso 4/4] KB...")
        try:
            
            generar_kb(DIRECTORIO_SALIDA_MD, ARCHIVO_LEMATIZADO, ARCHIVO_SALIDA_CLUSTERS, 'lemas_problema')
        except Exception as e:
            print(f"Error Clustering: {e}")
            sys.exit(1)
    else:
        print("[Paso 4/4] KB incidencias. Se usará archivos existentes.")


    
    if args.rag or israg:
        print("\n[Paso 4/5] KB...")
        try:
            
            rag(DIRECTORIO_SALIDA_MD, DIRECTORIO_CHROMA_DB, 'lemas_problema')
        except Exception as e:
            print(f"Error Clustering: {e}")
            sys.exit(1)
    else:
        print("[Paso 4/5] KB incidencias. Se usará archivos existentes.")


    

    #procesar_incidencias_a_csv("./incidencias", "incidencias_consolidadas.csv")
    #iniciar()

    

    # Aquí iría la lógica real de análisis de frecuencias.

if __name__ == '__main__':
    main()