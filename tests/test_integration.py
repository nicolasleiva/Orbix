import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
from src.prediction import predict_trajectory
from src.optimization import OrbitalPathOptimizer
from src.quantum_alerts import QuantumCollisionAlertSystem
from src.telemetry import TelemetryMonitor
from src.graph_analysis import build_satellite_graph

class TestIntegration(unittest.TestCase):
    """Pruebas de integración para verificar la interacción entre componentes."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Inicializar componentes
        self.optimizer = OrbitalPathOptimizer()
        self.alert_system = QuantumCollisionAlertSystem()
        self.telemetry = TelemetryMonitor()
        
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
        self.tle_data = {
            "satellite_id": "12345",
            "data": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        self.trajectory_df = pd.DataFrame({
            "x": np.linspace(950, 1100, 5),
            "y": np.linspace(1950, 2100, 5),
            "z": np.linspace(2950, 3100, 5)
        })
    
    @patch('src.prediction.global_model')
    def test_prediction_to_collision_alert(self, mock_model):
        """Prueba la integración entre predicción y alertas de colisión."""
        # Configurar el mock para la predicción
        mock_output = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        mock_model.return_value = mock_output
        
        # Realizar la predicción
        prediction_result = predict_trajectory(self.tle_data)
        
        # Verificar que la predicción tiene el formato esperado
        self.assertEqual(prediction_result["satellite_id"], "12345")
        self.assertIn("trajectory", prediction_result)
        
        # Convertir la predicción a un DataFrame para el sistema de alertas
        # (Simulando la conversión que se haría en un flujo real)
        trajectory_points = []
        for batch in prediction_result["trajectory"]:
            for point in batch:
                trajectory_points.append({"x": point[0], "y": point[1], "z": point[2]})
        
        trajectory_df = pd.DataFrame(trajectory_points)
        
        # Calcular la probabilidad de colisión
        collision_prob = self.alert_system.calculate_collision_probability(trajectory_df)
        
        # Verificar que la probabilidad está en el rango esperado
        self.assertGreaterEqual(collision_prob, 0.0)
        self.assertLessEqual(collision_prob, 1.0)
    
    def test_graph_to_optimization(self):
        """Prueba la integración entre análisis de grafos y optimización de rutas."""
        # Construir el grafo
        graph = build_satellite_graph(self.satellite_data, self.debris_data)
        
        # Verificar que el grafo se construyó correctamente
        self.assertIsNotNone(graph)
        
        # Usar los mismos datos para optimizar una ruta
        satellite_state = {"position": (950.0, 1950.0, 2950.0)}
        optimized_route = self.optimizer.optimize_route(satellite_state, self.debris_data)
        
        # Verificar que se obtuvo una ruta
        self.assertIsInstance(optimized_route, list)
    
    def test_optimization_to_telemetry(self):
        """Prueba la integración entre optimización y telemetría."""
        # Optimizar una ruta
        satellite_state = {"position": (950.0, 1950.0, 2950.0)}
        optimized_route = self.optimizer.optimize_route(satellite_state, self.debris_data)
        
        # Convertir la ruta a un formato compatible con telemetría
        # (Simulando la conversión que se haría en un flujo real)
        predictions = tf.constant([[point[0], point[1], point[2]] for point in optimized_route])
        ground_truth = tf.constant([[point[0]+0.1, point[1]+0.1, point[2]+0.1] for point in optimized_route])
        
        # Actualizar métricas de telemetría
        self.telemetry.update_metrics(predictions, ground_truth)
        
        # Obtener métricas
        metrics = self.telemetry.get_prometheus_metrics()
        
        # Verificar que las métricas están presentes
        self.assertIn('position_error', metrics)
        self.assertIn('velocity_error', metrics)

if __name__ == '__main__':
    unittest.main()