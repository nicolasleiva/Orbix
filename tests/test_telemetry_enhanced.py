import unittest
import tensorflow as tf
import time
from unittest.mock import patch, MagicMock
from src.telemetry_enhanced import TelemetryMonitor

class TestTelemetryEnhanced(unittest.TestCase):
    """Pruebas unitarias para el módulo de telemetría mejorado."""
    
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
        
        # Verificar que se actualizó la latencia
        self.assertGreater(updated_values['prediction_latency'], 0)
    
    def test_update_collision_probability(self):
        """Prueba que la probabilidad de colisión se actualice correctamente."""
        # Valor inicial
        initial_value = self.telemetry.metrics['collision_probability'].result().numpy()
        
        # Actualizar probabilidad
        test_probability = 0.75
        self.telemetry.update_collision_probability(test_probability)
        
        # Verificar que el valor ha cambiado
        updated_value = self.telemetry.metrics['collision_probability'].result().numpy()
        self.assertNotEqual(initial_value, updated_value)
        self.assertEqual(updated_value, test_probability)
    
    def test_update_optimization_metrics(self):
        """Prueba que las métricas de optimización se actualicen correctamente."""
        # Valores iniciales
        initial_values = {
            'optimization_time': self.telemetry.metrics['optimization_time'].result().numpy(),
            'fuel_consumption_estimate': self.telemetry.metrics['fuel_consumption_estimate'].result().numpy(),
            'trajectory_deviation': self.telemetry.metrics['trajectory_deviation'].result().numpy(),
            'orbital_stability': self.telemetry.metrics['orbital_stability'].result().numpy()
        }
        
        # Actualizar métricas
        self.telemetry.update_optimization_metrics(
            optimization_time=150.5,
            fuel_consumption=25.3,
            trajectory_deviation=0.15,
            orbital_stability=0.85
        )
        
        # Verificar que los valores han cambiado
        self.assertNotEqual(
            initial_values['optimization_time'],
            self.telemetry.metrics['optimization_time'].result().numpy()
        )
        self.assertNotEqual(
            initial_values['fuel_consumption_estimate'],
            self.telemetry.metrics['fuel_consumption_estimate'].result().numpy()
        )
        self.assertNotEqual(
            initial_values['trajectory_deviation'],
            self.telemetry.metrics['trajectory_deviation'].result().numpy()
        )
        self.assertNotEqual(
            initial_values['orbital_stability'],
            self.telemetry.metrics['orbital_stability'].result().numpy()
        )
    
    def test_get_prometheus_metrics(self):
        """Prueba que se obtengan correctamente las métricas para Prometheus."""
        # Actualizar métricas primero
        self.telemetry.update_metrics(self.predictions, self.ground_truth)
        
        # Obtener métricas
        metrics = self.telemetry.get_prometheus_metrics()
        
        # Verificar que todas las métricas estén presentes
        expected_metrics = [
            'position_error', 'velocity_error', 'collision_probability',
            'prediction_latency', 'optimization_time', 'fuel_consumption_estimate',
            'trajectory_deviation', 'orbital_stability'
        ]
        
        for metric_name in expected_metrics:
            self.assertIn(metric_name, metrics)
            self.assertIsInstance(metrics[metric_name], float)
    
    @patch('src.telemetry_enhanced.push_to_gateway')
    def test_export_to_prometheus(self, mock_push_to_gateway):
        """Prueba que la exportación a Prometheus funcione correctamente."""
        # Actualizar métricas primero
        self.telemetry.update_metrics(self.predictions, self.ground_truth)
        
        # Llamar al método
        self.telemetry.export_to_prometheus('http://localhost:9091')
        
        # Verificar que se llamó a push_to_gateway
        mock_push_to_gateway.assert_called_once()
        args, kwargs = mock_push_to_gateway.call_args
        
        # Verificar los argumentos
        self.assertEqual(args[0], 'http://localhost:9091')
        self.assertEqual(kwargs['job'], 'orbix_satellite_metrics')
    
    def test_reset_metrics(self):
        """Prueba que las métricas se reinicien correctamente."""
        # Actualizar métricas primero
        self.telemetry.update_metrics(self.predictions, self.ground_truth)
        
        # Obtener valores antes del reinicio
        before_reset = {name: metric.result().numpy() 
                       for name, metric in self.telemetry.metrics.items()}
        
        # Reiniciar métricas
        self.telemetry.reset_metrics()
        
        # Obtener valores después del reinicio
        after_reset = {name: metric.result().numpy() 
                      for name, metric in self.telemetry.metrics.items()}
        
        # Verificar que los valores han cambiado a cero o valores iniciales
        for name in ['position_error', 'velocity_error', 'collision_probability']:
            self.assertNotEqual(before_reset[name], after_reset[name])
            # Las métricas de error pueden reiniciarse a NaN o 0
            self.assertTrue(after_reset[name] == 0 or np.isnan(after_reset[name]))
    
    def test_get_metrics_summary(self):
        """Prueba que el resumen de métricas se genere correctamente."""
        # Actualizar métricas primero
        self.telemetry.update_metrics(self.predictions, self.ground_truth)
        
        # Obtener resumen
        summary = self.telemetry.get_metrics_summary()
        
        # Verificar que el resumen es una cadena no vacía
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        
        # Verificar que contiene información de todas las métricas
        for name in self.telemetry.metrics.keys():
            self.assertIn(name, summary)

if __name__ == '__main__':
    unittest.main()