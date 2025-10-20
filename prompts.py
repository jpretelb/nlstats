

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