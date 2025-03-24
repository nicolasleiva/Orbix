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

# Configuración del sistema de alertas cuánticas
QUANTUM_SIMULATOR_TYPE = os.getenv('QUANTUM_SIMULATOR_TYPE', 'vqe')  # Opciones: 'vqe', 'grover', 'basic'
QUANTUM_SHOTS = int(os.getenv('QUANTUM_SHOTS', '1000'))  # Número de mediciones en simulación cuántica
QUANTUM_NOISE_MODEL = os.getenv('QUANTUM_NOISE_MODEL', 'low')  # Opciones: 'low', 'high'

# Configuración de la API SSC (Space Science Center)
SSC_API_URL = os.getenv('SSC_API_URL', 'https://sscweb.gsfc.nasa.gov/WS/sscr/2')
SSC_API_KEY = os.getenv('SSC_API_KEY', 'YVZMezqqb5Zf3HQ2Tz6G3Ckdni04nDTXJZrrQNq2')

# Configuración de la API de SpaceX
SPACEX_API_URL = os.getenv('SPACEX_API_URL', 'https://api.spacexdata.com/v3')

# Configuración de la API de NOAA
NOAA_TOKEN = os.getenv('NOAA_TOKEN', 'CFgjhxTTdGlPhzzRymcXRZIeasOovcdf')
