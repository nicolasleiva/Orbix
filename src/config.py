import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SatelliteWazePro")

# Configuración de Kafka
KAFKA_CONFIG = {
    "bootstrap.servers": "kafka-cluster:9092",
    "group.id": "satellite-waze",
    "auto.offset.reset": "latest"
}

# Ruta del modelo (ajusta según corresponda)
MODEL_PATH = "models/production/v1"