import configparser
import os
from typing import Dict

def load_db_config(db_name: str) -> Dict[str, str]:
    """
    Carga la configuración de la base de datos especificada desde el archivo config.ini.
    
    db_name debe ser 'mssql' o 'postgres'.
    Devuelve un diccionario con 'server', 'port', 'database', 'username'.
    """
    config = configparser.ConfigParser()
    
    # Intenta leer el archivo de configuración
    config_file = 'config.ini'
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"El archivo de configuración '{config_file}' no se encuentra. Por favor, créalo.")
    
    config.read(config_file)
    
    if db_name not in config:
        raise ValueError(f"La sección [{db_name}] no se encuentra en el archivo de configuración.")
        
    # Extrae solo los campos necesarios (las claves se pasan por CLI)
    db_info = {
        'server': config.get(db_name, 'server'),
        'port': config.get(db_name, 'port'),
        'database': config.get(db_name, 'database'),
        'username': config.get(db_name, 'username')
    }
    
    return db_info

def load_gemini_token() -> str:
    """
    Carga el token de Gemini desde la variable de entorno GEMINI_API_KEY.
    """
    token = os.getenv("GEMINI_API_KEY")
    if not token:
        raise ValueError(
            "El token de Gemini no se ha encontrado. "
            "Por favor, define la variable de entorno 'GEMINI_API_KEY'."
        )
    return token
# Ejemplo de uso (opcional, solo para verificar)
if __name__ == '__main__':
    try:
        mssql_conf = load_db_config('mssql')
        print("Configuración MSSQL cargada:", mssql_conf)
    except Exception as e:
        print(f"Error al cargar configuración: {e}")