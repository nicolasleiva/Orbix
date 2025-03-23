import os
import logging
from dotenv import load_dotenv

# Cargar variables de entorno desde archivo .env
load_dotenv()

# Configuración de logging
log_level = os.getenv('LOG_LEVEL', 'INFO')
log_file = os.getenv('LOG_FILE', 'logs/orbix.log')

# Asegurar que el directorio de logs existe
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Configuración avanzada de logging
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Orbix")

# Configuración de Kafka
KAFKA_CONFIG = {
    "bootstrap.servers": os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka-cluster:9092'),
    "group.id": os.getenv('KAFKA_GROUP_ID', 'satellite-waze'),
    "auto.offset.reset": os.getenv('KAFKA_AUTO_OFFSET_RESET', 'latest')
}

# Configuración de la API
API_HOST = os.getenv('API_HOST', '0.0.0.0')
API_PORT = int(os.getenv('API_PORT', '8000'))
API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'

# Ruta del modelo
MODEL_PATH = os.getenv('MODEL_PATH', 'models/production/v1')
