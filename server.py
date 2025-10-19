import os
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
import json
from pydantic import BaseModel
from bs4 import BeautifulSoup
from fastapi.middleware.cors import CORSMiddleware # <-- 1. AÑADIR ESTE IMPORT
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
from fastapi.responses import StreamingResponse
import io
import matplotlib.pyplot as plt

import secrets # <-- 1. AÑADIR ESTE IMPORT
from fastapi import FastAPI, HTTPException, Depends, status # <-- 2. AÑADIR Imports de Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials # <-- 3. AÑADIR Imports de seguridad
from fastapi.responses import HTMLResponse
from fastapi.responses import JSONResponse

# --- CARGA DE CONFIGURACIÓN Y CLIENTES (SE EJECUTA UNA SOLA VEZ AL INICIAR) ---
# --- MODELOS DE DATOS Y APLICACIÓN FASTAPI ---
class QueryRequest(BaseModel):
    pregunta: str

plt.switch_backend('Agg')

app = FastAPI(
    title="API de Base de Conocimiento ERP",
    description="Una API para consultar soluciones a incidencias del ERP NISIRA."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)

# --- FUNCIONES AUXILIARES ---
def limpiar_html(texto_html: str) -> str:
    """Elimina etiquetas HTML de una cadena de texto."""
    soup = BeautifulSoup(texto_html, "lxml")
    return soup.get_text(separator=" ", strip=True)

def crear_prompt_estructurado(contexto: str, pregunta_usuario: str) -> str:
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



security = HTTPBasic() # Esquema de Basic Auth

# Credenciales fijas y seguras (usando 'secrets.compare_digest' para evitar ataques de temporización)
USUARIO_CORRECTO = secrets.compare_digest(b"admin", b"admin")
CLAVE_CORRECTA = secrets.compare_digest(b"clave123", b"clave123")

def verificar_credenciales(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Función de dependencia para Basic Authentication.
    Compara las credenciales recibidas con las credenciales fijas.
    """
    current_username_bytes = credentials.username.encode("utf8")
    current_password_bytes = credentials.password.encode("utf8")
    
    # Usar secrets.compare_digest para comparar cadenas sin exponer la duración del proceso
    is_correct_username = secrets.compare_digest(current_username_bytes, b"admin")
    is_correct_password = secrets.compare_digest(current_password_bytes, b"clave123")

    if not (is_correct_username and is_correct_password):
        # Levanta una excepción HTTP 401 para Basic Auth
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credenciales incorrectas",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username # Retorna el nombre de usuario si es correcto

@app.get("/status", tags=["Salud"])
def get_status():
    """
    Endpoint de chequeo de salud (Health Check)
    Devuelve un simple OK sin requerir autenticación.
    """
    return {"status": "OK"}
    
@app.get("/test.html", response_class=HTMLResponse, tags=["Test"])
def servir_test_html(username: str = Depends(verificar_credenciales)):
    """
    Sirve el archivo test.html. Requiere Basic Auth.
    """
    try:
        # Asume que test.html está en la misma carpeta
        with open("./html/test.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        # Si el archivo no existe
        raise HTTPException(status_code=404, detail="El archivo test.html no fue encontrado.")
    except Exception as e:
        # Otros errores de lectura
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {e}")
# -----------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, tags=["index"])
def servir_test_html(username: str = Depends(verificar_credenciales)):
    
    try:
        # Asume que test.html está en la misma carpeta
        with open("./html/index.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        return html_content
    except FileNotFoundError:
        # Si el archivo no existe
        raise HTTPException(status_code=404, detail="El archivo test.html no fue encontrado.")
    except Exception as e:
        # Otros errores de lectura
        raise HTTPException(status_code=500, detail=f"Error al leer el archivo: {e}")
# -----------------------------------------------------------------------

@app.post("/consultar-kb")
async def consultar_base_conocimiento(request: QueryRequest):
    """
    Recibe una pregunta, la limpia, y devuelve una respuesta estructurada.
    """
    # --- INICIALIZACIÓN Y CONEXIÓN (se ejecuta en cada llamada) ---
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return {"error": "GEMINI_API_KEY no configurada."}
    genai.configure(api_key=api_key)

    client = chromadb.PersistentClient(path='knowledge_base_db')
    collections = client.list_collections() 
    
    print("All collections in the DB:")
    if collections:
        # Imprimir solo los nombres para una vista más limpia
        for col in collections:
            print(f"- {col.name}")

    embedding_function = chromadb.utils.embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
    collection = client.get_collection(name='lemas_problema', embedding_function=embedding_function)
    
    #model = genai.GenerativeModel('gemini-1.5-flash-latest')
    model = genai.GenerativeModel('gemini-2.5-flash')
    # --- FIN DE LA INICIALIZACIÓN ---

    pregunta_html = request.pregunta
    pregunta_limpia = limpiar_html(pregunta_html)
    
    results = collection.query(
        query_texts=[pregunta_limpia],
        n_results=15
    )
    
    contexto = "\n---\n".join(results['documents'][0])
    metadatos = results['metadatas'][0]

    prompt = crear_prompt_estructurado(contexto, pregunta_limpia)
    print(prompt)
    try:
        response = model.generate_content(prompt)
        
        # Devuelve directamente el texto Markdown generado por el modelo
        # dentro de una clave JSON.
        respuesta_texto = response.text.replace('```markdown', '').replace('```', '').strip()
        
        return {"respuesta_markdown": respuesta_texto}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la respuesta del modelo: {e}")

@app.get("/get-grupos")
async def consultar_base_conocimiento():
    csv_file = "incidencias_consolidadas.csv"

    df_incidencias_cluster = pd.read_csv(csv_file)
    grupos = df_incidencias_cluster['modulo'].unique().tolist()

    
    try:
        
        return {"grupos": grupos}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar la respuesta del modelo: {e}")

@app.get("/img-cluster/{modulo}", tags=["Imágenes"])
async def obtener_imagen_modulo(modulo: str):
    img_name = modulo.replace(' ', '_').replace('-', '_') + ".png"

    carpeta = Path("./imgs/cluster/")

    ruta_archivo =  carpeta / img_name
    
    print(ruta_archivo)
    if not ruta_archivo.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"La imagen '{img_name}' no fue encontrada"
        )
    return FileResponse(path=ruta_archivo, filename=img_name)
 
@app.get("/img-codo/{modulo}", tags=["Imágenes"])
async def obtener_imagen_modulo(modulo: str):
    img_name = modulo.replace(' ', '_').replace('-', '_') + ".png"

    carpeta = Path("./imgs/codo/")

    ruta_archivo =  carpeta / img_name
    
    print(ruta_archivo)
    if not ruta_archivo.is_file():
        raise HTTPException(
            status_code=404, 
            detail=f"La imagen '{img_name}' no fue encontrada"
        )
    return FileResponse(path=ruta_archivo, filename=img_name)

@app.get("/reporte/{modulo}", tags=["Reportes Dinámicos"])
async def generar_reporte_modulo(modulo: str, tipo: str):
    df = load_and_merge_data(modulo)

    print(df)
    
    if tipo == 'tema':
        # Reporte 1: Incidencias por ia_theme (Cantidad)
        data_plot = df.groupby('ia_theme').size().sort_values(ascending=True)
        title = f"Incidencias por Tema (Módulo: {modulo})"
        xlabel = "Cantidad de Incidencias"
        ylabel = "Tema Descriptivo (ia_theme)"
        
    elif tipo == 'empresa':
        # Reporte 2: Incidencias por empresa (Cantidad)
        data_plot = df.groupby('empresa').size().sort_values(ascending=True)
        title = f"Incidencias por Empresa (Módulo: {modulo})"
        xlabel = "Cantidad de Incidencias"
        ylabel = "Empresa"
        
    else:
        raise HTTPException(
            status_code=400, 
            detail="El parámetro 'tipo' debe ser 'tema' o 'empresa'."
        )

    # ----------------------------------------
    # GENERACIÓN DEL GRÁFICO (Matplotlib)
    # ----------------------------------------
    
    fig, ax = plt.subplots(figsize=(10, len(data_plot) * 0.5 + 1), dpi=100) # Tamaño dinámico
    data_plot.plot(kind='barh', ax=ax, color='#1f77b4') # Gráfico de barras horizontal

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    
    # Ajustar el layout para evitar cortes de etiquetas largas
    plt.tight_layout() 
    
    # ----------------------------------------
    # CONVERSIÓN A IMAGEN EN MEMORIA
    # ----------------------------------------
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)  # Liberar memoria de Matplotlib
    buf.seek(0)
    
    # Devolver la imagen como respuesta de streaming
    return StreamingResponse(buf, media_type="image/png")


@app.get("/reporte/{modulo}/mes_tema", tags=["Reportes Dinámicos"])
async def reporte_incidencias_mensuales_por_tema(modulo: str):
    """Genera un gráfico de barras apiladas mostrando la tendencia mensual de incidencias por tema ('ia_theme')."""
    
    df = load_and_merge_data(modulo)
    
    # Preparar datos para agrupación mensual
    df['mes'] = df['fecha'].dt.to_period('M').astype(str)
    
    # Agrupación: Contar incidencias por Mes y por Tema (ia_theme)
    pivot_data = df.groupby(['mes', 'ia_theme']).size().unstack(fill_value=0)
    
    # Ordenar cronológicamente
    pivot_data.index = pd.to_datetime(pivot_data.index)
    pivot_data = pivot_data.sort_index()
    
    return generate_plot(
        pivot_data,
        title=f"Tendencia Mensual de Incidencias por Tema (Módulo: {modulo})",
        xlabel="Mes",
        ylabel="Cantidad de Incidencias",
        kind='stacked_bar'
    )
@app.get("/reporte/{modulo}/pareto", tags=["Reportes Dinámicos"])
async def reporte_pareto_incidencias(modulo: str):
    """
    Genera el gráfico de Pareto (80/20) para identificar el 20% de temas que
    causan el 80% de las incidencias.
    """
    
    df = load_and_merge_data(modulo)
    
    return generate_pareto_plot(df, modulo)

def generate_plot(data_plot: pd.Series, title: str, xlabel: str, ylabel: str, kind: str = 'barh', **kwargs) -> StreamingResponse:
    """Genera el gráfico en memoria (PNG) y lo devuelve como StreamingResponse."""
    
    if kind == 'barh':
        # Gráfico de barras horizontal (para temas/empresas)
        fig, ax = plt.subplots(figsize=(10, len(data_plot) * 0.5 + 1), dpi=100)
        data_plot.plot(kind='barh', ax=ax, color='#1f77b4', **kwargs)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        plt.tight_layout()
        
    elif kind == 'stacked_bar':
        # Gráfico de barras apiladas (para mes_tema)
        num_months = len(data_plot.index)
        fig, ax = plt.subplots(figsize=(1.2 * num_months + 2, 8), dpi=100)
        data_plot.plot(kind='bar', stacked=True, ax=ax, **kwargs)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        ax.tick_params(axis='x', rotation=45) 
        ax.legend(title='Tema', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar para la leyenda
        
    # Conversión a bytes en memoria
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

def generate_pareto_plot(data: pd.DataFrame, modulo: str) -> StreamingResponse:
    """Genera el gráfico de Pareto (80/20) en memoria (PNG) y lo devuelve."""
    
    # 1. Preparar los datos
    theme_counts = data.groupby('ia_theme').size().sort_values(ascending=False)
    total_incidencias = theme_counts.sum()
    
    # 2. Calcular Frecuencia Relativa y Acumulada
    df_pareto = pd.DataFrame(theme_counts, columns=['Count'])
    df_pareto['Percentage'] = (df_pareto['Count'] / total_incidencias) * 100
    df_pareto['Cumulative Percentage'] = df_pareto['Percentage'].cumsum()
    
    # 3. GENERACIÓN DEL GRÁFICO
    fig, ax1 = plt.subplots(figsize=(12, 6), dpi=100)
    
    # Eje Y Primario (Barras de Conteo)
    color = '#1f77b4'
    ax1.bar(df_pareto.index, df_pareto['Count'], color=color)
    ax1.set_xlabel("Tema de Incidencia", fontsize=12)
    ax1.set_ylabel("Cantidad de Incidencias", color=color, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticklabels(df_pareto.index, rotation=45, ha='right')
    
    # Eje Y Secundario (Línea de Porcentaje Acumulado)
    ax2 = ax1.twinx()
    color = '#ff7f0e'
    ax2.plot(df_pareto.index, df_pareto['Cumulative Percentage'], color=color, marker='o', linestyle='--')
    ax2.set_ylabel("Porcentaje Acumulado", color=color, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 105) # Escala de 0% a 105%
    
    # Línea de referencia del 80%
    ax2.axhline(80, color='red', linestyle=':', linewidth=1.5, label='80% Acumulado')
    ax2.legend(loc='center right')
    
    # Título
    plt.title(f"Análisis 80/20 (Pareto) de Incidencias por Tema: {modulo}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # 4. CONVERSIÓN A IMAGEN EN MEMORIA
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    return StreamingResponse(buf, media_type="image/png")

def load_and_merge_data(modulo: str) -> pd.DataFrame:
    """Carga, filtra y une los tres DataFrames necesarios."""
    
    # Normalizar el nombre del módulo para usar en rutas de archivo
    # modulo_safe = urllib.parse.unquote(modulo)

    modulo_file = modulo.replace(' ', '_').replace('-', '_') + ""

    
    # Rutas base
    path_theme = Path("./themes/")
    path_lemma = Path("./lemma/")
    path_cluster = Path("./cluster/")
    

    try:
        # DataFrame 1: Temas (Cluster -> ia_theme)
        df1 = pd.read_csv(path_theme / f"{modulo_file}.csv", usecols=["cluster", "ia_theme"])
        df1 = df1.rename(columns={"cluster": "cluster_consolidado"})
        
        # DataFrame 2: Cluster por Incidencia (idincidencia -> cluster_consolidado)
        df2 = pd.read_csv(path_cluster / f"{modulo_file}.csv", usecols=["idincidencia", "cluster_consolidado"])
        
        # DataFrame 3: Metadatos de Incidencia (idincidencia -> empresa, fecha)
        # Reutilizamos el mismo archivo de lemma, pero con diferentes columnas
        df3 = pd.read_csv(path_lemma / f"{modulo_file}.csv", usecols=["idincidencia", "empresa", "fechacreacion"])
        
        print(11)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Archivo CSV no encontrado para el módulo '{modulo_file}': {e.name}"
        )

    print(f"DataFrames cargados para el módulo '{modulo}':")
    # 1. Merge df2 (incidencias-cluster) con df3 (incidencias-meta)
    df_merged = pd.merge(df2, df3, on="idincidencia", how="inner")
    
    # 2. Merge con df1 (temas)
    df_final = pd.merge(df_merged, df1, on="cluster_consolidado", how="left")
    
    # Convertir 'fecha' a datetime si es necesario para futuros análisis
    df_final['fecha'] = pd.to_datetime(df_final['fechacreacion'], errors='coerce')

    return df_final

@app.get("/data/{modulo}/pareto", tags=["Datos Tabulares"])
async def data_pareto_incidencias(modulo: str):
    """
    Devuelve los datos del Análisis de Pareto (80/20) en formato JSON.
    Columnas: tema, cantidad, porcentaje, porcentaje_acumulado.
    """
    
    df = load_and_merge_data(modulo)
    
    # 1. Conteo y ordenación (el más representativo primero)
    theme_counts = df.groupby('ia_theme').size().sort_values(ascending=False)
    total_incidencias = theme_counts.sum()
    
    # 2. Creación del DataFrame de Pareto
    df_pareto = pd.DataFrame(theme_counts, columns=['cantidad'])
    
    # 3. Cálculo de porcentajes
    df_pareto['porcentaje'] = (df_pareto['cantidad'] / total_incidencias) * 100
    df_pareto['porcentaje_acumulado'] = df_pareto['porcentaje'].cumsum()
    
    # 4. Formateo de nombres y valores
    df_pareto = df_pareto.reset_index().rename(columns={'ia_theme': 'tema'})
    
    # Redondear porcentajes a dos decimales
    df_pareto['porcentaje'] = df_pareto['porcentaje'].round(2)
    df_pareto['porcentaje_acumulado'] = df_pareto['porcentaje_acumulado'].round(2)
    
    # 5. Conversión a formato de lista de diccionarios (JSON)
    data_json = df_pareto.to_dict('records')
    
    # Devolver la tabla de datos en JSON
    return JSONResponse(content=data_json)

# --- Bloque para ejecución directa (opcional, para testing) ---
if __name__ == "__main__":
    import uvicorn
    print("Iniciando servidor Uvicorn para pruebas...")
    uvicorn.run(app, host="127.0.0.1", port=8000)


