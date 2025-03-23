import logging
import numpy as np
import pandas as pd
from ortools.constraint_solver import routing_enums_pb2, pywrapcp

logger = logging.getLogger("SatelliteWazePro")

class OrbitalPathOptimizer:
    """
    Optimiza la ruta orbital utilizando OR-Tools aplicando un TSP sobre puntos de desechos.
    A partir de la posición actual del satélite y las coordenadas (x, y, z) de cada desecho,
    se obtiene la ruta óptima que minimiza la distancia total a recorrer.
    """
    def optimize_route(self, satellite_state: dict, debris_data: pd.DataFrame):
        """
        Optimiza la ruta a partir del estado del satélite y datos de desechos.
        Se espera que debris_data tenga columnas 'debris_id', 'x', 'y', 'z'.
        satellite_state debe tener una llave 'position' con una tupla (x, y, z).

        Retorna una lista de tuplas (debris_id, x, y, z) en el orden óptimo.
        """
        if debris_data.empty or 'position' not in satellite_state:
            logger.error("Datos de desechos vacíos o estado del satélite sin 'position'.")
            return []

        # Posición del satélite (punto de partida)
        start = np.array(satellite_state['position'])
        # Extraer posiciones (x, y, z) de cada desecho
        points = debris_data[['x','y','z']].to_numpy()

        # Crear matriz de todos los puntos (incluye el punto de inicio)
        all_points = np.concatenate(([start], points), axis=0)
        num_points = len(all_points)
        
        # Construir la matriz de distancias
        distance_matrix = np.zeros((num_points, num_points))
        for i in range(num_points):
            for j in range(num_points):
                if i != j:
                    distance_matrix[i][j] = np.linalg.norm(all_points[i] - all_points[j])
                else:
                    distance_matrix[i][j] = 0
        
        # Configurar el modelo de OR-Tools para TSP (sin retorno al punto de inicio)
        manager = pywrapcp.RoutingIndexManager(num_points, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(distance_matrix[from_node][to_node] * 1000)  # Convertir a enteros

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Parámetros de solución
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC

        solution = routing.SolveWithParameters(search_parameters)
        if solution:
            index = routing.Start(0)
            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index > 0:  # Omitir el nodo de inicio
                    debris_id = debris_data.iloc[node_index - 1]['debris_id']
                    x, y, z = all_points[node_index]
                    route.append((debris_id, x, y, z))
                index = solution.Value(routing.NextVar(index))
            logger.info("Ruta optimizada: %s", route)
            return route
        else:
            logger.error("No se encontró solución en la optimización.")
            return []
