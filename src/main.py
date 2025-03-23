import sys
import uvicorn
import logging
import pandas as pd
from .config import logger
from .prediction import load_model
from .api import create_app
from .graph_analysis import build_satellite_graph, analyze_graph
from .optimization import OrbitalPathOptimizer
from .quantum_alerts import QuantumCollisionAlertSystem

def main():
    # Cargar el modelo Transformer para trayectorias
    logger.info("Cargando modelo Transformer...")
    try:
        load_model()
    except Exception as e:
        logger.error("Error crítico al cargar el modelo: %s", str(e))
        sys.exit(1)
    
    # Ejemplo de análisis en grafos
    sample_satellite_data = pd.DataFrame({
        "satellite_id": [101, 102],
        "feature": [0.5, 0.8]
    })
    sample_debris_data = pd.DataFrame({
        "debris_id": [201, 202],
        "feature": [0.3, 0.7],
        "x": [1000, 1050],
        "y": [2000, 2050],
        "z": [3000, 3050]
    })
    graph = build_satellite_graph(sample_satellite_data, sample_debris_data)
    analyze_graph(graph)
    
    # Ejemplo de optimización de ruta orbital
    optimizer = OrbitalPathOptimizer()
    satellite_state = {"position": (950.0, 1950.0, 2950.0)}
    optimized_route = optimizer.optimize_route(satellite_state, sample_debris_data)
    logger.info("Ruta optimizada obtenida: %s", optimized_route)
    
    # Ejemplo de cálculo de probabilidad de colisión a partir de una trayectoria dummy
    import numpy as np
    # Simular una trayectoria con 5 puntos (x, y, z)
    trajectory_df = pd.DataFrame({
        "x": np.linspace(950, 1100, 5),
        "y": np.linspace(1950, 2100, 5),
        "z": np.linspace(2950, 3100, 5)
    })
    alert_system = QuantumCollisionAlertSystem()
    collision_prob = alert_system.calculate_collision_probability(trajectory_df)
    logger.info("Probabilidad de colisión calculada: %f", collision_prob)
    
    # Iniciar el servidor API (FastAPI)
    logger.info("Iniciando servidor API en el puerto 8000...")
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
