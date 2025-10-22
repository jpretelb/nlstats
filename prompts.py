

def crear_prompt_consulta_inc(contexto: str, pregunta_usuario: str) -> str:
    """
    Crea el prompt final para el modelo, incluyendo instrucciones, un ejemplo (few-shot),
    el contexto de la base de conocimiento y la pregunta del usuario.
    """
    prompt_final = f"""
Eres un asistente experto en el ERP NISIRA. Tu tarea es responder la pregunta del usuario basándote únicamente en el siguiente contexto proporcionado.
Tu única salida debe ser el código Markdown. No incluyas explicaciones adicionales antes o después del código Markdown.
Debes generar tu respuesta final en un formato MARKDOWN estricto, con la siguiente estructura: Resumen, Incidencias (donde se detallará cada incidencia encontrada en el contexto).

---
## EJEMPLO PRÁCTICO (FEW-SHOT)

### INPUT (Pregunta del usuario):
"Error en el cálculo de la gratificación."

### OUTPUT (Markdown Esperado):
```markdown
## Resumen

El cálculo de la gratificación del mes de diciembre no coincide con los días ingresados: 30 faltas y 1 día de licencia con goce. El sistema proyecta 30 días para el cálculo de la gratificación, sin tener en cuenta los días de falta o licencia.

## Incidencias

### Incidencia 246753

#### Problema
El cálculo de la gratificación del mes de diciembre no coincide con los días ingresados: 30 faltas y 1 día de licencia con goce. El sistema proyecta 30 días para el cálculo de la gratificación, sin tener en cuenta los días de falta o licencia.

#### Solución
El sistema proyecta 30 días para el cálculo de la gratificación en diciembre. Para corregir el cálculo, se deben ingresar días de ajuste negativo en noviembre para que se consideren en el cálculo de días computables.

### Incidencia 246803

#### Problema
El cálculo de la gratificación para el trabajador considera un básico de 1539.99 en lugar de 1600. El básico de 1539.99 se ingresó manualmente en noviembre debido a un aumento salarial el 7 de noviembre. NISIRA no calcula el básico proporcional para los meses con cambios salariales. El cálculo debería usar el básico de diciembre, ya que la gratificación abarca de julio a diciembre.

#### Solución
Se cambió temporalmente el mes base del cálculo de la gratificación a diciembre. Se recomienda verificar el cálculo (sin reprocesar la gratificación, ya que el mes base volvió a noviembre según norma).


### Incidencia 246990

#### Problema
Error en el cálculo de la gratificación. No se considera el ingreso por vacaciones en el periodo de cálculo, por lo que la gratificación se calcula descontando esos días.

#### Solución
Se activó el check de afecto a provisión del concepto de adelanto de vacaciones. Verificar y confirmar solución.
```

---

## Contexto Proporcionado:
{contexto}
## Pregunta del Usuario:
{pregunta_usuario}

## Tu Respuesta (solo en formato Markdown):
"""

    return prompt_final











def crear_prompt_consulta_resumen(contexto: str, pregunta_usuario: str) -> str:
    
    prompt_final = f"""
Eres un asistente experto en el ERP NISIRA. Tu tarea es responder la pregunta del usuario basándote únicamente en el siguiente contexto proporcionado.
Tu única salida debe ser el código Markdown. No incluyas explicaciones adicionales antes o después del código Markdown.
Debes generar tu respuesta final en un formato MARKDOWN estricto, con la siguiente estructura: Resumen, Posibles soluciones (donde se detallará las posibles solucions).

---
## EJEMPLO PRÁCTICO (FEW-SHOT)

### INPUT (Pregunta del usuario):
"Error en el cálculo de la gratificación."

### OUTPUT (Markdown Esperado):
```markdown
## Resumen

El cálculo de la gratificación del mes de diciembre no coincide con los días ingresados: 30 faltas y 1 día de licencia con goce. El sistema proyecta 30 días para el cálculo de la gratificación, sin tener en cuenta los días de falta o licencia.

## Posibles soluciones

Podría revisar la formula de días computables, y compararlos contra los reportes de dias trabajados.
Segun las incidencias *246753* y *246753* se puede corregir los dias computables agregando un concepto adicional.
```

---

## Contexto Proporcionado:
{contexto}
## Pregunta del Usuario:
{pregunta_usuario}

## Tu Respuesta (solo en formato Markdown):
"""

    return prompt_final












def crear_prompt_estructurado2(contexto: str, pregunta_usuario: str) -> str:
    """
    Crea el prompt final para el modelo, incluyendo instrucciones, un ejemplo (few-shot),
    el contexto de la base de conocimiento y la pregunta del usuario.
    """
    prompt_final = f"""
Eres un asistente experto en el ERP NISIRA. Tu tarea es responder la pregunta del usuario basándote únicamente en el siguiente contexto proporcionado.
Tu única salida debe ser el código Markdown. No incluyas explicaciones adicionales antes o después del código Markdown.
Debes generar tu respuesta final en un formato MARKDOWN estricto, sientete en la libertad de agregar algo más siempre que este dentro del contexto.

---

## Contexto Proporcionado:
{contexto}
## Pregunta del Usuario:
{pregunta_usuario}

## Tu Respuesta (solo en formato Markdown):
"""

    return prompt_final



def crear_prompt_consulta_resumen(texto_completo_incidencia):
    """Crea el prompt estructurado para la IA."""

    instruccion = """## ROL Y OBJETIVO (CRÍTICO)
Eres un **Analista de Datos experto en Procesamiento de Lenguaje Natural (NLP)** con enfoque en la **estandarización de lemas (tokens)**.
Tu misión es procesar y limpiar incidencias de soporte del software NISIRA ERP. El objetivo supremo es crear un **corpus de texto limpio, consistente y estandarizado**, donde **cada concepto clave sea un token único y en singular**.
---
## TAREA PRINCIPAL
Analiza la siguiente incidencia, **extrae la información clave y reescríbela de forma normalizada en un JSON estricto**, aplicando **RIGUROSAMENTE** las reglas de estandarización.


---
## REGLAS DE NORMALIZACIÓN
1.  **ESTRATEGIA SINGLE WORD (CRÍTICO):** Normaliza siempre los nombres completos y sus abreviaturas a la sigla o token único oficial. El objetivo es que cada concepto sea una sola palabra. Por ejemplo:
    * "Orden de Servicio" -> **"OSC"**
    * "Orden de Compra" -> **"OCO"**
    * "Guía de remisión" -> **"GRM"**
    * "Nota de credito" -> **"NCR"**
    * "Compensación por tiempo de servicio" -> **"CTS"**
    * "Liquidacion de Beneficios sociales" o similar -> **"Liquidaciones"**
    * "Orden de Venta" -> **"OVT"**
    * "Renta de quinta categoria" o "renta 5ta" -> **"R5TA"**
    * "Recursos Humanos" o "Nómina" -> **"RRHH"**
    * "Packing List" -> **"Packing"**
    * "Packing List" -> **"Packing"**
    * "Packing List" -> **"Packing"**
    * "Cuentas por Pagar", "Cuentas/pagar" -> **"Cuentas_por_Pagar"**
    * "Cuentas por Cobrar", "Cuentas-x-Cobrar" -> **"Cuentas_por_Cobrar"**
    * "Base de Datos" -> **"BD"**

2.  **FORMA SINGULAR (CRÍTICO):** Convierte siempre los sustantivos plurales a su forma singular. El objetivo es unificar el mismo concepto en una sola palabra, puedes colocar entre parentesis la palabra varios.
    * **Ejemplo:** "Las facturas no coinciden" -> se convierte en "La factura no coincide (varios)".
    * **Ejemplo:** "Errores en las órdenes de compra" -> se convierte en "Error en la orden de compra (varios)".
    * **Ejemplo:** "Formulario de liquidaciones" -> se convierte en "Formulario de liquidacion".
    * **Ejemplo:** "Documento de liquidaciones" -> se convierte en "Documento de liquidacion".

3.  **FECHAS Y PERIODOS:** Normaliza todas las fechas y periodos al formato **mes año** (ej: `enero 2024`) o **dd mes año** (ej: `31 enero 2024`). Usa siempre el nombre del mes completo y en minúsculas.
    * **Ejemplo:** "31/01/2024" -> **"31 enero 2024"**
    * **Ejemplo:** "enero 2024", "01-2024", "202401" -> **"enero 2024"**
    * **Ejemplo:** "periodo 2024-2025" -> **"periodo 2024 2025"**


4.  **ANONIMIZACIÓN:**
    * Reemplaza nombres de personas por la palabra **"el trabajador"** (solo si es relevante).
    * Reemplaza nombres o siglas de empresas clientes (ej: "GMHBERRIES", "GMH") por la frase **"la empresa"**.

5.  **LIMPIEZA GENERAL:**
    * Elimina frases de cortesía ("hola", "gracias"), información personal (teléfonos, emails), caracteres especiales, comillas y saltos de línea.
    * Reescribe el problema y la solución de forma clara y concisa.

6.  **RESPUESTA CLARA (respuesta_clara):**
    * La solución debe tener algún evento o cambio que el agente haya realizado.
    * Habilitar un parametro, cambiar un procedimiento almancenado, configuraciones, o si se indica que se ha corregido con el usuario.
    * Si solo hay una confirmación, por ejemplo: "esta corregido", "se confirma que se corrigió", en estos casos no se indica una respuesta o solución clara.



---
## FORMATO DE SALIDA (JSON)
El resultado debe ser un JSON estricto con estas claves. Si no hay solución, el valor debe ser `null`.
* `problema`: (string) Descripción normalizada.
* `solucion`: (string|null) Descripción normalizada.
* `categoria`: (string) La categoría más apropiada.
* `urgencia_estimada`: (string) "Alta", "Media" o "Baja".
* `respuesta_clara`: (string) "SI", "NO", Si hay una respuesta muy ambigua, por ejemplo "se solucionó", "se confirma la solucion", "se corrigió" o simplemente no hay respuesta colocar "NO", si la respuesta tiene un detalle de la solucion dada, colocar "SI".
* `entidades`: (array de objetos) Cada objeto con `{"tipo": "...", "nombre": "..."}`.

---
## CONTEXTO Y DICCIONARIO
* **ERP:** NISIRA ERP
* **Módulos Principales:** Ventas, Compras, Almacén, Contabilidad, Activo Fijo, Bancos, Cuentas_por_Cobrar, Cuentas_por_Pagar, RRHH, Producción, Mantenimiento.
* **Siglas Principales:** PLE, SIRE, PLAME, EDOC, SUNAT.
* **Tipos de Entidades:** "Módulo", "Proceso", "Documento", "Entidad Bancaria", "Concepto Contable", "Entidad Fiscal".

---
## EJEMPLO PRÁCTICO (FEW-SHOT)

### INPUT (Texto de la incidencia):
"hola amigos de GMHBERRIES, tengo un problm con la renta de 5ta cat. para el empleado Juan Perez en la Planilla Mensual de Pagos de enero 2024, el calculo del impuesto no me cuadra con el reporte del nisira erp, además tengo problemas con el calulo de liquidaciones y otros beneficios sociales. Me sale error."

### OUTPUT (JSON Esperado):
```json
{
    "problema": "El cálculo del impuesto R5TA para el trabajador en PLAME del periodo enero 2024 es incorrecto o no coincide con los montos del reporte generado por el sistema NISIRA ERP en la empresa. Y tiene problema con el cálculo de la liquidacion (varios) y beneficio social.",
    "solucion": "Se corrigió las formulas necesarias en el catalogo de conceptos",
    "respuesta_clara": "SI",
    "categoria": "Cálculo de Nómina / RRHH",
    "urgencia_estimada": "Alta",
    "entidades": [
        {
            "tipo": "Módulo",
            "nombre": "RRHH"
        },
        {
            "tipo": "Concepto Contable",
            "nombre": "R5TA"
        },
        {
            "tipo": "Documento",
            "nombre": "PLAME"
        }
    ]
}"""
    
    prompt = f"{instruccion}\n\nConversación de la Incidencia:\n---\n{texto_completo_incidencia}\n---\n\nJSON de Salida:"
    return prompt

def crear_prompt_thema(texto_completo_incidencia):
    
    instruccion = """## ROL Y OBJETIVO
Eres un Analista de Datos experto en Procesamiento de Lenguaje Natural (NLP). Tu misión es procesar incidencias de soporte 
del software NISIRA ERP para crear temas por cada cluster que te envio.

---
## TAREA PRINCIPAL
Analiza cada cluster y colocale un titulo o tema representativo.

## REDACCION
La redaccion debe ser corta, evitar el uso de emojis.

## TERMINOS CLAVE
Estos terminos fueron reducidos, pero debes tener en cuenta las siguientes equivalencias para una mejor interpretacion:
R5TA : Renta de quita categoria

## FORMATO DE SALIDA (JSON)
El resultado debe ser un JSON estricto con estas claves.
* `cluster_id`: (integer) Número de cluster.
* `tema`: (string) Thema encontrado.

### OUTPUT (JSON Esperado):
```json
    [
        {"cluster_id": 0, "tema": "Errores en Cálculo de Impuestos"},
        {"cluster_id": 1, "tema": "Problemas con Reportes Financieros"},
        {"cluster_id": 2, "tema": "Dificultades en Gestión de Inventarios"}
    ]```"""
    
    prompt = f"{instruccion}\n\nClusters encontrados:\n---\n{texto_completo_incidencia}\n---\n\nJSON de Salida:"
    return prompt
