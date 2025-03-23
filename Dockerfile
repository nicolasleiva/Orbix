FROM python:3.9-slim

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requisitos primero para aprovechar la caché de Docker
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Crear directorio para logs
RUN mkdir -p logs

# Crear directorio para modelos
RUN mkdir -p models/production/v1

# Exponer el puerto para la API
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["python", "-m", "src.main"]