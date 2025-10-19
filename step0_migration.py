import pyodbc
from datetime import datetime, timedelta
import psycopg2
from psycopg2 import Error
import json
from tqdm import tqdm

def run_migration(mssql_conf, postgres_conf, mssql_pass, postgres_pass, start_date_str, end_date_str):
    """
    Función que realiza la migración real.
    
    Recibe todos los parámetros necesarios (configuración y claves).
    """
    DATE_FORMAT = "%Y%m%d"

    start_date = datetime.strptime(start_date_str, DATE_FORMAT).date()
    end_date = datetime.strptime(end_date_str, DATE_FORMAT).date()
    
    if start_date > end_date:
        raise ValueError("La fecha de inicio (start_date) no puede ser posterior a la fecha final (end_date).")
    
    delta = timedelta(days=1)

    local_conn = local_connection(postgres_conf['server'], postgres_conf['port'], postgres_conf['database'], postgres_conf['username'], postgres_pass)
    source_conn = source_connection(mssql_conf['server'], mssql_conf['port'], mssql_conf['database'], mssql_conf['username'], mssql_pass)

    current_date = start_date
    while current_date <= end_date:
        # Aquí se ejecuta la acción que necesites para cada fecha
        
        # Opcional: Volver a formatear a string si lo necesitas (por ejemplo, para logs o nombres de archivos)
        date_for_processing = current_date.strftime(DATE_FORMAT)
        
        #print(f"Procesando fecha: {date_for_processing}")
        
        ids_incidencias = return_incidencias_fecha_origen(conn = local_conn, dia = current_date)
        ids_respuestas = return_respuestas_fecha_origen(conn = local_conn, dia = current_date)

        #print(ids_incidencias)
        idsjson = json.dumps(ids_incidencias)
        idsjson_r = json.dumps(ids_respuestas)
        #print(idsjson)

        incidencias = incidencias_origen(source_conn, date_for_processing, idsjson)

        

        #for incidencia in incidencias:
        for incidencia in tqdm(incidencias, desc="Insertando Incidencias del " + date_for_processing):
            idincidencia = incidencia[0]
            titulo = incidencia[1]
            contenido = incidencia[2]
            fecha_inc = incidencia[3]
            modulo = incidencia[4]
            prioridad = incidencia[5]
            usuario = incidencia[6]
            empresa = incidencia[7]
            
            if ids_incidencias.count(idincidencia) != 0:
                #print(f"La incidencia {idincidencia} ya existe en la base de datos destino. Saltando inserción.")
                continue

            exito = insertar_incidencia(
                conn=local_conn,
                id_incidencia=idincidencia,
                titulo=titulo,
                contenido=contenido,
                modulo=modulo,
                empresa=empresa,
                fecha_creacion=fecha_inc
            )

            #if exito:
            #    print(f"Incidencia {idincidencia} insertada correctamente.")
            #else:
            #    print(f"Error al insertar incidencia {idincidencia}.")
            
        respuestas = respuestas_origen(source_conn, date_for_processing, idsjson_r)
        for rpt in tqdm(respuestas, desc="Insertando respuestas del " + date_for_processing):
            idincidencia = rpt[0]
            idrespuesta = rpt[1]
            usuario = rpt[2]
            respuesta = rpt[3]
            tipousuario = rpt[4]
            visibilidad = rpt[5]
            fechacreacion = rpt[6]

            if ids_respuestas.count(idrespuesta) != 0:
                #print(f"La incidencia {idincidencia} ya existe en la base de datos destino. Saltando inserción.")
                continue
            
            exito = insertar_respuesta(
                conn=local_conn,
                id_respuesta=idrespuesta,
                id_incidencia=idincidencia,
                respuesta=respuesta,
                tipousuario=tipousuario,
                visibilidad=visibilidad,
                fecha_creacion=fechacreacion
            )

        
        # 4. Avanzar al siguiente día
        current_date += delta

        
    
    
    if local_conn:
        local_conn.close()
    if source_conn:
        source_conn.close()




    print("\n--- INICIANDO LÓGICA DE MIGRACIÓN REAL ---")
    print(f"Fuente (MSSQL): {mssql_conf['database']} en {mssql_conf['server']}")
    print(f"Destino (Postgres): {postgres_conf['database']} en {postgres_conf['server']}")
    
    # Aquí iría la lógica de conexión con librerías como:
    # 1. Establecer conexión a MSSQL usando mssql_pass.
    # 2. Leer datos.
    # 3. Establecer conexión a Postgres usando postgres_pass.
    # 4. Escribir datos.
    
    print("Conexiones y transferencia simuladas... Proceso completado.")
    return True # Devolver un estado de éxito


def crear_incidencia(conn, idincidencia, incidencia, contenido, fecha_inc, modulo, prioridad, usuario, empresa):
    cursor = conn.cursor()
    rows = cursor.execute("EXEC VSSP_CREAR_INCIDENCIA ?, ?, ?, ?, ?, ?, ?, ?", idincidencia, incidencia, contenido, fecha_inc, modulo, prioridad, usuario, empresa)
    cursor.commit()
    cursor.close()
    return rows

def incidencias_origen(conn, dia, json_ids):
    cursor = conn.cursor()
    rows =cursor.execute("exec NSP_RETURN_INC_FECHA ?, ?",dia,json_ids).fetchall()
    cursor.close()

    return rows

def respuestas_origen(conn, dia, json_ids):
    cursor = conn.cursor()
    rows = cursor.execute("exec NSP_RETURN_RESP_FECHA ?, ?",dia, json_ids).fetchall()
    cursor.close()
    return rows

def return_respuestas_fecha_origen(conn, dia):
    cursor = conn.cursor()
    query_func = f"SELECT * FROM get_respuestas_fecha('{dia}');"
    cursor.execute(query_func)
    ids = []
    
    for row in cursor.fetchall():
        ids.append(row[0])
    return ids

def return_incidencias_fecha_origen(conn, dia):
    cursor = conn.cursor()
    query_func = f"SELECT * FROM get_incidencias_fecha('{dia}');"
    cursor.execute(query_func)
    ids = []
    
    for row in cursor.fetchall():
        ids.append(row[0])
    return ids

def insertar_incidencia(
    conn,
    id_incidencia: int,
    titulo: str,
    contenido: str,
    modulo: str,
    empresa: str,
    fecha_creacion: datetime = None
) -> bool:
    """
    Inserta una nueva incidencia en la tabla 'incidencia'.

    :param id_incidencia: El ID único de la incidencia (INTEGER PRIMARY KEY).
    :param titulo: Título de la incidencia (TEXT).
    :param contenido: Contenido/descripción detallada (TEXT).
    :param modulo: Módulo del sistema afectado (VARCHAR).
    :param empresa: Nombre de la empresa (VARCHAR).
    :param fecha_creacion: Fecha de creación. Si es None, usa la fecha y hora actual.
    :return: True si la inserción fue exitosa, False en caso contrario.
    """

    sql_insert = """
    INSERT INTO incidencia (
        id, 
        titulo, 
        contenido, 
        fechacreacion, 
        modulo, 
        empresa
    ) 
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    
    # Parámetros para la consulta
    params = (
        id_incidencia,
        titulo,
        contenido,
        fecha_creacion,
        modulo,
        empresa
    )

    try:
        cursor = conn.cursor()
        
        cursor.execute(sql_insert, params)
        
        conn.commit()
        #print(f"Incidencia con ID {id_incidencia} insertada exitosamente.")
        return True
    
    except pyodbc.Error as ex:
        # En caso de error, intenta hacer rollback si la conexión existe
        if conn:
            conn.rollback()
        sqlstate = ex.args[0]
        #print(f"Error al insertar incidencia: {sqlstate}")
        print(ex)
        return False



def insertar_respuesta(
    conn,
    id_respuesta: int,
    id_incidencia: int,
    respuesta: str,
    tipousuario: str,
    visibilidad: str,
    fecha_creacion: datetime
) -> bool:
    """
    Inserta una nueva incidencia en la tabla 'respuesta'.
    :param id_respuesta: El ID único de la respuesta (INTEGER PRIMARY KEY).
    :param id_incidencia: El ID de la incidencia asociada (FOREIGN KEY).
    :param respuesta: Contenido de la respuesta (TEXT).
    :param tipousuario: Tipo de usuario que responde (VARCHAR).
    :param visibilidad: Visibilidad de la respuesta (VARCHAR).
    :param fecha_creacion: Fecha de creación.
    :return: True si la inserción fue exitosa, False en caso contrario.
    """
    

    sql_insert = """
    INSERT INTO respuesta (
        id,
        idincidencia, 
        respuesta,
        tipousuario,
        visibilidad,
        fechacreacion
    ) 
    VALUES (%s, %s, %s, %s, %s, %s);
    """
    
    # Parámetros para la consulta
    params = (
        id_respuesta,
        id_incidencia,
        respuesta,
        tipousuario,
        visibilidad,
        fecha_creacion
    )

    try:
        cursor = conn.cursor()
        
        cursor.execute(sql_insert, params)
        
        conn.commit()
        #print(f"Incidencia con ID {id_incidencia} insertada exitosamente.")
        return True
    
    except pyodbc.Error as ex:
        # En caso de error, intenta hacer rollback si la conexión existe
        if conn:
            conn.rollback()
        sqlstate = ex.args[0]
        #print(f"Error al insertar incidencia: {sqlstate}")
        print(ex)
        return False
    


def source_connection(srv, port, db, usr, passwd):
    #SERVER = '10.64.5.20,49880'
    #SERVER = '10.0.0.204,53714'
    #DATABASE = 'GestionIncAct'
    #USERNAME = 'nisira'
    #USERNAME = 'portalnis'
    #PASSWORD = 'O3e1jSoZi5wh'

    connectionString = f'DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={srv},{port};DATABASE={db};UID={usr};PWD={passwd};TrustServerCertificate=yes;'
    print(connectionString)
    conn = pyodbc.connect(connectionString)
    return conn


def local_connection(srv, port, db, usr, passwd):
    #SERVER = '10.64.5.20,49880'
    #SERVER = '10.0.0.204,53714'
    #DATABASE = 'GestionIncAct'
    #USERNAME = 'nisira'
    #USERNAME = 'portalnis'
    #PASSWORD = 'O3e1jSoZi5wh'
    conn = psycopg2.connect(
        user=usr,
        password=passwd,
        host=srv,
        port=port,
        database=db
    )
    return conn
