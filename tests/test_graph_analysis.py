import unittest
import pandas as pd
import networkx as nx
from unittest.mock import patch, MagicMock
from src.graph_analysis import build_satellite_graph, analyze_graph

class TestGraphAnalysis(unittest.TestCase):
    """Pruebas unitarias para el módulo de análisis de grafos."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Datos de prueba
        self.satellite_data = pd.DataFrame({
            "satellite_id": [101, 102],
            "feature": [0.5, 0.8]
        })
        self.debris_data = pd.DataFrame({
            "debris_id": [201, 202],
            "feature": [0.3, 0.7],
            "x": [1000, 1050],
            "y": [2000, 2050],
            "z": [3000, 3050]
        })
    
    def test_build_satellite_graph_valid_data(self):
        """Prueba que el grafo se construya correctamente con datos válidos."""
        # Construir el grafo
        graph = build_satellite_graph(self.satellite_data, self.debris_data)
        
        # Verificar que el grafo no es None
        self.assertIsNotNone(graph)
        
        # Verificar que el grafo tiene el número correcto de nodos
        self.assertEqual(graph.number_of_nodes(), 4)  # 2 satélites + 2 desechos
        
        # Verificar que el grafo tiene el número correcto de aristas
        self.assertEqual(graph.number_of_edges(), 2)  # Mínimo entre len(satellite_data) y len(debris_data)
    
    def test_build_satellite_graph_empty_data(self):
        """Prueba que el grafo maneje correctamente datos vacíos."""
        # Datos de entrada vacíos
        empty_satellite_data = pd.DataFrame()
        empty_debris_data = pd.DataFrame()
        
        # Construir el grafo con datos vacíos
        graph = build_satellite_graph(empty_satellite_data, empty_debris_data)
        
        # Verificar que el grafo es None
        self.assertIsNone(graph)
    
    @patch('src.graph_analysis.plt')
    @patch('src.graph_analysis.nx.spring_layout')
    def test_analyze_graph(self, mock_spring_layout, mock_plt):
        """Prueba que el análisis del grafo funcione correctamente."""
        # Construir el grafo
        graph = build_satellite_graph(self.satellite_data, self.debris_data)
        
        # Configurar el mock para nx.spring_layout
        mock_pos = {node: [0, 0] for node in range(4)}
        mock_spring_layout.return_value = mock_pos
        
        # Ejecutar la función
        analyze_graph(graph)
        
        # Verificar que se llamaron los métodos esperados
        mock_plt.figure.assert_called_once()
        mock_plt.title.assert_called_once()
        mock_plt.show.assert_called_once()
    
    def test_analyze_graph_none(self):
        """Prueba que el análisis maneje correctamente un grafo None."""
        # Ejecutar la función con un grafo None
        analyze_graph(None)
        
        # No hay aserciones específicas, solo verificamos que no haya excepciones

if __name__ == '__main__':
    unittest.main()