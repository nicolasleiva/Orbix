import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configurar path para importar módulos del proyecto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.quantum_api_integrator import QuantumApiIntegrator
from src.config import logger

# Configurar logging
logging.basicConfig(level=logging.INFO)

def main():
    """
    Ejemplo de uso del integrador de APIs cuánticas para generar alertas de colisión
    utilizando datos reales de satélites.
    """
    logger.info("Iniciando ejemplo de integración de APIs cuánticas")
    
    # Inicializar el integrador
    integrador = QuantumApiIntegrator()
    
    # Definir ventana de tiempo para datos históricos (últimas 24 horas)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)
    
    # Ejemplo 1: Obtener datos de un satélite específico
    satellite_id = "25544"  # ISS (Estación Espacial Internacional)
    logger.info(f"Obteniendo datos para el satélite {satellite_id}")
    
    satellite_data = integrador.get_satellite_data(
        satellite_id=satellite_id,
        start_time=start_time,
        end_time=end_time
    )
    
    # Convertir a formato de trayectoria
    trajectory = integrador.convert_api_data_to_trajectory(satellite_data)
    if not trajectory.empty:
        logger.info(f"Trayectoria obtenida con {len(trajectory)} puntos")
        print("Primeros puntos de la trayectoria:")
        print(trajectory.head())
    else:
        logger.warning("No se pudo obtener trayectoria válida")
    
    # Ejemplo 2: Predecir trayectoria futura
    logger.info(f"Prediciendo trayectoria futura para el satélite {satellite_id}")
    predicted_trajectory = integrador.predict_trajectory(
        satellite_id=satellite_id,
        start_time=start_time,
        end_time=end_time,
        prediction_hours=48  # Predecir 48 horas en el futuro
    )
    
    if not predicted_trajectory.empty:
        logger.info(f"Trayectoria predicha con {len(predicted_trajectory)} puntos")
        print("Primeros puntos de la trayectoria predicha:")
        print(predicted_trajectory.head())
    else:
        logger.warning("No se pudo predecir trayectoria")
    
    # Ejemplo 3: Generar alerta de colisión
    logger.info(f"Generando alerta de colisión para el satélite {satellite_id}")
    alert = integrador.generate_collision_alert(
        satellite_id=satellite_id,
        prediction_hours=72  # Analizar riesgo para las próximas 72 horas
    )
    
    if "error" not in alert:
        logger.info(f"Alerta generada con ID: {alert.get('alert_id')}")
        print("\nDetalles de la alerta:")
        print(f"Nivel de alerta: {alert.get('alert_level')}")
        print(f"Probabilidad de colisión: {alert.get('collision_probability'):.4f}")
        print(f"Tiempo hasta máximo acercamiento: {alert.get('time_to_closest_approach')}")
        print(f"Acciones recomendadas: {alert.get('recommended_actions')}")
    else:
        logger.error(f"Error al generar alerta: {alert.get('error')}")
    
    # Ejemplo 4: Analizar múltiples satélites
    satellite_ids = ["25544", "43013", "48274"]  # ISS, TESS, Starlink-1654
    logger.info(f"Analizando colisiones entre múltiples satélites: {satellite_ids}")
    
    alerts = integrador.analyze_multiple_satellites(
        satellite_ids=satellite_ids,
        prediction_hours=48
    )
    
    if alerts and "error" not in alerts[0]:
        logger.info(f"Se generaron {len(alerts)} alertas")
        for i, alert in enumerate(alerts):
            print(f"\nAlerta {i+1}:")
            print(f"Satélites: {alert.get('satellite_id')} - {alert.get('other_object_id')}")
            print(f"Nivel: {alert.get('alert_level')}")
            print(f"Probabilidad: {alert.get('collision_probability'):.4f}")
    else:
        logger.warning("No se generaron alertas para los satélites analizados")

if __name__ == "__main__":
    main()