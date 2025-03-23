import unittest
import tensorflow as tf
from unittest.mock import patch, MagicMock
from src.telemetry import TelemetryMonitor

class TestTelemetry(unittest.TestCase):
    """Pruebas unitarias para el módulo de telemetría."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.telemetry = TelemetryMonitor()
        
        # Datos de prueba
        self.predictions = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.ground_truth = tf.constant([[1.1, 2.1, 3.1], [4.1, 5.1, 6.1]])
    
    def test_update_metrics(self):
        """Prueba que las métricas se actualicen correctamente."""
        # Guardar los valores iniciales
        initial_values = {name: metric.result().numpy() 
                         for name, metric in self.telemetry.metrics.items()}
        
        # Actualizar métricas
        self.telemetry.update_metrics(self.predictions, self.ground_truth)
        
        # Verificar que los valores han cambiado
        updated_values = {name: metric.result().numpy() 
                         for name, metric in self.telemetry.metrics.items()}
        
        # Al menos position_error y velocity_error deberían cambiar
        self.assertNotEqual(initial_values['position_error'], 
                           updated_values['position_error'])
        self.assertNotEqual(initial_values['velocity_error'], 
                           updated_values['velocity_error'])
    
    def test_get_prometheus_metrics(self):
        """Prueba que se obtengan correctamente las métricas para Prometheus."""
        # Actualizar métricas primero
        self.telemetry.update_metrics(self.predictions, self.ground_truth)
        
        # Obtener métricas
        metrics = self.telemetry.get_prometheus_metrics()
        
        # Verificar que todas las métricas estén presentes
        self.assertIn('position_error', metrics)
        self.assertIn('velocity_error', metrics)
        self.assertIn('collision_probability', metrics)
        self.assertIn('prediction_latency', metrics)
        self.assertIn('optimization_time', metrics)
        
        # Verificar que los valores son números
        for value in metrics.values():
            self.assertIsInstance(value, float)
    
    @patch('src.telemetry.TelemetryMonitor.get_prometheus_metrics')
    def test_export_to_prometheus(self, mock_get_metrics):
        """Prueba que la exportación a Prometheus funcione correctamente."""
        # Configurar el mock
        mock_metrics = {
            'position_error': 0.1,
            'velocity_error': 0.2,
            'collision_probability': 0.05,
            'prediction_latency': 10.5,
            'optimization_time': 5.3
        }
        mock_get_metrics.return_value = mock_metrics
        
        # Llamar al método (actualmente es un stub)
        self.telemetry.export_to_prometheus('http://localhost:9091')
        
        # Verificar que se llamó al método para obtener métricas
        mock_get_metrics.assert_called_once()

if __name__ == '__main__':
    unittest.main()