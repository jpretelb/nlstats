-- ##################################################
-- CREACIÓN DE TABLAS
-- ##################################################

-- TABLA INCIDENCIA
-- SERIAL proporciona una secuencia auto-incremental para la clave primaria.
-- TIMESTAMP WITHOUT TIME ZONE (o TIMESTAMP) almacena fecha y hora.
CREATE TABLE incidencia (
    id INTEGER PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    contenido TEXT,
    fechacreacion TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    modulo VARCHAR(200),
    empresa VARCHAR(200)
);

-- TABLA RESPUESTA
-- idincidencia es una clave foránea que enlaza con la tabla 'incidencia'.
CREATE TABLE respuesta (
    id INTEGER PRIMARY KEY,
    idincidencia INTEGER NOT NULL,
    respuesta TEXT,
    tipousuario VARCHAR(200),
    visibilidad VARCHAR(200),
    fechacreacion TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);


-- ##################################################
-- CREACIÓN DE FUNCIONES
-- ##################################################

-- Función 1: Obtener Incidencias por Fecha
-- Retorna todas las incidencias creadas en una fecha específica (xfecha).
CREATE OR REPLACE FUNCTION get_incidencias_fecha(xfecha DATE)
RETURNS TABLE (
    id INTEGER ,
    titulo VARCHAR,
    contenido TEXT,
    fechacreacion TIMESTAMP WITHOUT TIME ZONE,
    modulo VARCHAR,
    empresa VARCHAR
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        i.id,
        i.titulo,
        i.contenido,
        i.fechacreacion,
        i.modulo,
        i.empresa
    FROM
        incidencia i
    WHERE
        -- Convierte el TIMESTAMP a DATE para comparar solo la fecha.
        i.fechacreacion::DATE = xfecha;
END;
$$ LANGUAGE plpgsql;

-- Función 2: Obtener Respuestas por Fecha
-- Retorna todas las respuestas creadas en una fecha específica (xfecha).
CREATE OR REPLACE FUNCTION get_respuestas_fecha(xfecha DATE)
RETURNS TABLE (
    id INTEGER,
    idincidencia INTEGER,
    respuesta TEXT,
    tipousuario VARCHAR,
    visibilidad VARCHAR,
    fechacreacion TIMESTAMP WITHOUT TIME ZONE
)
AS $$
BEGIN
    RETURN QUERY
    SELECT
        r.id,
        r.idincidencia,
        r.respuesta,
        r.tipousuario,
        r.visibilidad,
        r.fechacreacion
    FROM
        respuesta r
    WHERE
        -- Convierte el TIMESTAMP a DATE para comparar solo la fecha.
        r.fechacreacion::DATE = xfecha;
END;
$$ LANGUAGE plpgsql;

-- ##################################################
-- Ejemplo opcional de datos iniciales
-- ##################################################

-- INSERT INTO incidencia (titulo, contenido) VALUES ('Error de login', 'El usuario no puede acceder...');
-- INSERT INTO respuesta (idincidencia, respuesta, tipousuario) VALUES (1, 'Problema resuelto en el servidor', 'Admin');