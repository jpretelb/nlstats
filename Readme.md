# Instalacion

El proyecto consta de dos componentes, una herramienta CLI, y otra basada en Servicios Web para visualización e integración.

## Herramienta CLI

    docker build -f Dockerfile.cli -t python-dev .

## Servicio WEB

    docker build -t "nlstat-service" -f Dockerfile.service .

# Entrenamiento
Crear variable de entorno con key de Gemini.

    GEMINI_API_KEY=AI.....
Ejecutar herramienta CLI, se debe compartir la carpeta de la aplicación, para que se puedan generar los recursos y compartirlis posteriormente con la web.

    docker run -it --rm -v $(pwd):/app -e "GEMINI_API_KEY=$GEMINI_API_KEY" python-dev bash
Dentro del contenedor usa ejecutar main.py y revisar la documentación del CLI.

    python main.py

# Ejecución
Ejecutar el contenedor y publicarlo en el puerto 8080.

    docker run -d -p 8000:8000 -e "GEMINI_API_KEY=$GEMINI_API_KEY" nlstat-service
