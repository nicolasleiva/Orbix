import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.optimization import OrbitalPathOptimizer

class TestOptimization(unittest.TestCase):
    """Pruebas unitarias para el módulo de optimización de rutas orbitales."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.optimizer = OrbitalPathOptimizer()
        
        # Datos de prueba
        self.satellite_state = {"position": (1000.0, 2000.0, 3000.0)}
        self.debris_data = pd.DataFrame({
            "debris_id": [101, 102, 103],
            "x": [1100.0, 1200.0, 1300.0],
            "y": [2100.0, 2200.0, 2300.0],
            "z": [3100.0, 3200.0, 3300.0]
        })
    
    @patch('src.optimization.pywrapcp.RoutingModel.SolveWithParameters')
    def test_optimize_route_success(self, mock_solve):
        """Prueba que la optimización de ruta funcione correctamente."""
        # Configurar el mock para simular una solución exitosa
        mock_solution = MagicMock()
        mock_solve.return_value = mock_solution
        
        # Configurar el comportamiento del mock para simular una ruta
        mock_solution.Value.return_value = 1  # Simular el siguiente nodo
        
        # Parchear el método IsEnd para controlar el flujo del bucle
        with patch('src.optimization.pywrapcp.RoutingModel.IsEnd', side_effect=[False, False, False, True]):
            # Parchear IndexToNode para simular los índices de nodos
            with patch('src.optimization.pywrapcp.RoutingIndexManager.IndexToNode', side_effect=[0, 1, 2, 3]):
                # Ejecutar la función
                result = self.optimizer.optimize_route(self.satellite_state, self.debris_data)
                
                # Verificar el resultado
                self.assertEqual(len(result), 2)  # Debería haber 2 puntos en la ruta (omitiendo el nodo inicial)
    
    def test_optimize_route_empty_data(self):
        """Prueba que la optimización maneje correctamente datos vacíos."""
        # Datos de entrada vacíos
        empty_debris_data = pd.DataFrame()
        
        # Ejecutar la función
        result = self.optimizer.optimize_route(self.satellite_state, empty_debris_data)
        
        # Verificar que se devuelve una lista vacía
        self.assertEqual(result, [])
    
    def test_optimize_route_missing_position(self):
        """Prueba que la optimización maneje correctamente un estado sin posición."""
        # Estado sin posición
        invalid_state = {"velocity": (1.0, 2.0, 3.0)}
        
        # Ejecutar la función
        result = self.optimizer.optimize_route(invalid_state, self.debris_data)
        
        # Verificar que se devuelve una lista vacía
        self.assertEqual(result, [])

if __name__ == '__main__':
    unittest.main()