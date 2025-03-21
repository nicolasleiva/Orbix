import sys
import uvicorn
import logging
import pandas as pd
from .config import logger
from .prediction import load_model
from .api import create_app
from .graph_analysis import build_satellite_graph, analyze_graph

def main():
    # Cargar el modelo Transformer para trayectorias
    logger.info("Cargando modelo Transformer...")
    try:
        load_model()
    except Exception as e:
        logger.error("Error crítico al cargar el modelo: %s", str(e))
        sys.exit(1)
    
    # (Opcional) Ejemplo de análisis en grafos:
    sample_satellite_data = pd.DataFrame({
        "satellite_id": [101, 102],
        "feature": [0.5, 0.8]
    })
    sample_debris_data = pd.DataFrame({
        "debris_id": [201, 202],
        "feature": [0.3, 0.7]
    })
    graph = build_satellite_graph(sample_satellite_data, sample_debris_data)
    analyze_graph(graph)
    
    # Iniciar el servidor API (FastAPI) en el puerto 8000
    logger.info("Iniciando servidor API en el puerto 8000...")
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
