FROM python:3.11-slim

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Crear directorio de trabajo
WORKDIR /app

# Copiar archivos necesarios
COPY ./src /app/src
COPY ./test /app/test
COPY requirements.txt /app/

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt
# Instalar Gradio directamente
RUN pip install --no-cache-dir gradio
# Exponer puerto para Gradio
EXPOSE 7860
# Configurar variables de entorno para Gradio
ENV GRADIO_SERVER_NAME="0.0.0.0"


# Comando por defecto
CMD ["python", "/app/src/main.py"]
