import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("SatelliteWazePro")

class QuantumCollisionAlertSystem:
    """
    Sistema de alertas que calcula una probabilidad de colisión
    de forma dummy usando la variación de distancias entre puntos de una trayectoria.
    """
    def calculate_collision_probability(self, trajectory: pd.DataFrame) -> float:
        """
        Calcula una probabilidad de colisión basada en las distancias entre puntos consecutivos.
        Se espera que `trajectory` contenga columnas ['x', 'y', 'z'].
        
        Retorna un valor entre 0 y 1.
        """
        if trajectory.empty:
            logger.error("Trayectoria vacía para el cálculo de colisión.")
            return 0.0
        
        coords = trajectory[['x', 'y', 'z']].to_numpy()
        if len(coords) < 2:
            logger.warning("Trayectoria insuficiente para calcular diferencias.")
            return 0.0
        
        # Calcular las distancias entre puntos consecutivos
        diffs = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        min_diff = np.min(diffs)
        # Fórmula dummy: si la distancia mínima es baja, aumenta el riesgo
        risk = 1 - (min_diff / (min_diff + 0.1))
        logger.info("Probabilidad de colisión calculada: %f", risk)
        return float(risk)
