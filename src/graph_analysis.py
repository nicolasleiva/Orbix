import pandas as pd
import logging
import matplotlib.pyplot as plt
import networkx as nx
from stellargraph import StellarGraph

logger = logging.getLogger("Orbix")

def build_satellite_graph(satellite_data: pd.DataFrame, debris_data: pd.DataFrame) -> StellarGraph:
    """
    Construye un grafo heterogéneo a partir de datos de satélites y desechos.
    Se espera que `satellite_data` y `debris_data` contengan las columnas 'satellite_id' y 'debris_id', respectivamente.
    """
    if satellite_data.empty or debris_data.empty:
        logger.warning("Datos insuficientes para construir el grafo.")
        return None

    satellite_nodes = satellite_data.set_index("satellite_id")
    debris_nodes = debris_data.set_index("debris_id")
    
    # Crear DataFrame de aristas ficticio basado en interacciones
    edges_df = pd.DataFrame({
        "source": satellite_data['satellite_id'].tolist(),
        "target": debris_data['debris_id'].tolist(),
        "weight": [1] * min(len(satellite_data), len(debris_data))
    })
    
    graph = StellarGraph(
        {"Satellite": satellite_nodes, "Debris": debris_nodes},
        edges={"interaction": edges_df}
    )
    return graph

def analyze_graph(graph: StellarGraph):
    """
    Analiza y visualiza el grafo satelital.
    """
    if graph is None:
        logger.error("No se proporcionó un grafo para analizar.")
        return
    logger.info("Número de nodos: %d", graph.number_of_nodes())
    logger.info("Número de aristas: %d", graph.number_of_edges())
    
    # Convertir a NetworkX para visualización
    nx_graph = graph.to_networkx()
    pos = nx.spring_layout(nx_graph)
    plt.figure(figsize=(8,6))
    nx.draw(nx_graph, pos, with_labels=True, node_size=500, font_size=10)
    plt.title("Visualización del grafo satelital")
    plt.show()
