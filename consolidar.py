import os
import json
import pandas as pd
#import spacy
#from tqdm import tqdm

from constants import PATH_INCIDENCIAS, CSV_CONSOLIDATE


def procesar_incidencias_a_csv():
    #tqdm.pandas()

    """
    Recorre una carpeta de archivos JSON de incidencias, los procesa
    y los consolida en un único archivo CSV aplanado.
    """
    lista_de_incidencias = []
    
    print(f"Buscando archivos JSON en la carpeta: '{PATH_INCIDENCIAS}'...")

    for nombre_archivo in os.listdir(PATH_INCIDENCIAS):
        if nombre_archivo.endswith('.json'):
            ruta_completa = os.path.join(PATH_INCIDENCIAS, nombre_archivo)
            
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
                        'respuesta_clara': data.get('respuesta_clara'),
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
        'idincidencia', 'respuesta_clara', 'fechacreacion', 'empresa', 'modulo', 'categoria', 
        'urgencia_estimada', 'problema', 'solucion', 'entidad_modulo', 
        'entidad_proceso', 'entidad_documento'
    ]
    df = df[columnas_ordenadas]

    df.to_csv(CSV_CONSOLIDATE, index=False, encoding='utf-8-sig')
    
    print("--------------------------------------------------")
    print(f"¡Proceso completado exitosamente!")
    print(f"Se ha creado el archivo '{CSV_CONSOLIDATE}' con {len(df)} filas.")
    print("--------------------------------------------------")