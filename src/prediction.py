import tensorflow as tf
import logging
from .quantum_trajectory import QuantumTrajectoryModel
from .config import MODEL_PATH

logger = logging.getLogger("SatelliteWazePro")

# Variable global para almacenar el modelo
global_model = None

def load_model():
    """
    Carga los pesos del modelo cuántico desde el directorio de producción.
    """
    global global_model
    global_model = QuantumTrajectoryModel()
    try:
        global_model.load_weights(MODEL_PATH)
        logger.info("Modelo cuántico cargado con éxito desde %s", MODEL_PATH)
    except Exception as e:
        logger.error("Error al cargar el modelo: %s", str(e))
        raise e

def predict_trajectory(tle_data: dict) -> dict:
    """
    Predice la trayectoria orbital usando el modelo cuántico a partir de datos TLE.
    Se espera que tle_data incluya claves 'data' y 'satellite_id'.
    """
    try:
        input_data = tle_data.get("data", [])
        if not input_data:
            raise ValueError("Datos TLE vacíos")
        input_tensor = tf.convert_to_tensor([input_data], dtype=tf.float32)
        prediction = global_model(input_tensor, training=False)
        return {
            "satellite_id": tle_data.get("satellite_id", "unknown"),
            "trajectory": prediction.numpy().tolist()
        }
    except Exception as e:
        logger.error("Error en la predicción: %s", str(e))
        return {"error": str(e)}
