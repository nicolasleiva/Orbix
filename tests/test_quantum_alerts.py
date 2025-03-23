import unittest
import pandas as pd
import numpy as np
import json
from datetime import datetime
from unittest.mock import patch, MagicMock
from src.quantum_alerts import QuantumCollisionAlertSystem

class TestQuantumAlerts(unittest.TestCase):
    """Pruebas unitarias para el sistema de alertas de colisión cuántica."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        # Parchear el productor de Kafka para evitar conexiones reales durante las pruebas
        self.producer_mock = MagicMock()
        with patch('confluent_kafka.Producer', return_value=self.producer_mock):
            self.alert_system = QuantumCollisionAlertSystem()
        
        # Datos de prueba - trayectoria con puntos que se acercan gradualmente
        self.trajectory_df = pd.DataFrame({
            "x": np.linspace(950, 1100, 5),
            "y": np.linspace(1950, 2100, 5),
            "z": np.linspace(2950, 3100, 5)
        })
        
        # Trayectoria con puntos que se acercan mucho (mayor riesgo)
        self.high_risk_trajectory = pd.DataFrame({
            "x": [950, 951, 952, 953, 954],
            "y": [1950, 1951, 1952, 1953, 1954],
            "z": [2950, 2951, 2952, 2953, 2954]
        })
        
        # Trayectoria con puntos muy separados (menor riesgo)
        self.low_risk_trajectory = pd.DataFrame({
            "x": [950, 1050, 1150, 1250, 1350],
            "y": [1950, 2050, 2150, 2250, 2350],
            "z": [2950, 3050, 3150, 3250, 3350]
        })
    
    def test_calculate_collision_probability_normal(self):
        """Prueba el cálculo de probabilidad de colisión con datos normales."""
        # Calcular la probabilidad
        probability = self.alert_system.calculate_collision_probability(self.trajectory_df)
        
        # Verificar que la probabilidad está en el rango [0, 1]
        self.assertGreaterEqual(probability, 0.0)
        self.assertLessEqual(probability, 1.0)
    
    def test_calculate_collision_probability_high_risk(self):
        """Prueba el cálculo de probabilidad de colisión con trayectoria de alto riesgo."""
        # Calcular la probabilidad
        probability = self.alert_system.calculate_collision_probability(self.high_risk_trajectory)
        
        # Verificar que la probabilidad es alta
        self.assertGreaterEqual(probability, 0.5)  # Esperamos un valor alto para trayectorias cercanas
    
    def test_calculate_collision_probability_low_risk(self):
        """Prueba el cálculo de probabilidad de colisión con trayectoria de bajo riesgo."""
        # Calcular la probabilidad
        probability = self.alert_system.calculate_collision_probability(self.low_risk_trajectory)
        
        # Verificar que la probabilidad es baja
        self.assertLessEqual(probability, 0.5)  # Esperamos un valor bajo para trayectorias separadas
    
    def test_calculate_collision_probability_empty(self):
        """Prueba el cálculo de probabilidad de colisión con datos vacíos."""
        # Datos de entrada vacíos
        empty_trajectory = pd.DataFrame()
        
        # Calcular la probabilidad
        probability = self.alert_system.calculate_collision_probability(empty_trajectory)
        
        # Verificar que la probabilidad es 0.0 para datos vacíos
        self.assertEqual(probability, 0.0)
    
    def test_calculate_collision_probability_single_point(self):
        """Prueba el cálculo de probabilidad de colisión con un solo punto."""
        # Datos con un solo punto
        single_point_trajectory = pd.DataFrame({
            "x": [950],
            "y": [1950],
            "z": [2950]
        })
        
        # Calcular la probabilidad
        probability = self.alert_system.calculate_collision_probability(single_point_trajectory)
        
        # Verificar que la probabilidad es 0.0 para un solo punto
        self.assertEqual(probability, 0.0)

    def test_generate_alert(self):
        """Prueba la generación de alertas con información completa."""
        # Generar una alerta
        alert = self.alert_system.generate_alert("SAT-001", self.trajectory_df)
        
        # Verificar que la alerta contiene todos los campos necesarios
        self.assertIn("alert_id", alert)
        self.assertIn("timestamp", alert)
        self.assertIn("satellite_id", alert)
        self.assertIn("collision_probability", alert)
        self.assertIn("alert_level", alert)
        self.assertIn("time_to_closest_approach", alert)
        self.assertIn("quantum_algorithm", alert)
        self.assertIn("recommended_actions", alert)
        
        # Verificar que el ID del satélite es correcto
        self.assertEqual(alert["satellite_id"], "SAT-001")
        
        # Verificar que la probabilidad está en el rango correcto
        self.assertGreaterEqual(alert["collision_probability"], 0.0)
        self.assertLessEqual(alert["collision_probability"], 1.0)
    
    def test_determine_alert_level(self):
        """Prueba la determinación del nivel de alerta basado en la probabilidad."""
        # Probar diferentes niveles de alerta
        self.assertEqual(self.alert_system._determine_alert_level(0.9), "CRÍTICO")
        self.assertEqual(self.alert_system._determine_alert_level(0.7), "ALTO")
        self.assertEqual(self.alert_system._determine_alert_level(0.3), "MEDIO")
        self.assertEqual(self.alert_system._determine_alert_level(0.1), "BAJO")
    
    def test_get_recommended_actions(self):
        """Prueba las recomendaciones basadas en el nivel de alerta."""
        # Verificar que cada nivel de alerta tiene recomendaciones apropiadas
        critico_actions = self.alert_system._get_recommended_actions("CRÍTICO")
        alto_actions = self.alert_system._get_recommended_actions("ALTO")
        medio_actions = self.alert_system._get_recommended_actions("MEDIO")
        bajo_actions = self.alert_system._get_recommended_actions("BAJO")
        
        # Verificar que hay recomendaciones para cada nivel
        self.assertTrue(len(critico_actions) > 0)
        self.assertTrue(len(alto_actions) > 0)
        self.assertTrue(len(medio_actions) > 0)
        self.assertTrue(len(bajo_actions) > 0)
        
        # Verificar que las recomendaciones críticas incluyen maniobra evasiva
        self.assertTrue(any("maniobra evasiva" in action.lower() for action in critico_actions))
    
    def test_apply_quantum_algorithm_vqe(self):
        """Prueba el algoritmo cuántico VQE."""
        # Configurar el sistema para usar VQE
        self.alert_system.quantum_simulator_type = "vqe"
        
        # Crear coordenadas de prueba
        coords = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        min_distance = 1.73  # Aproximadamente sqrt(3)
        
        # Calcular el riesgo con VQE
        risk = self.alert_system._apply_quantum_algorithm(coords, min_distance)
        
        # Verificar que el riesgo está en el rango correcto
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
    
    def test_apply_quantum_algorithm_grover(self):
        """Prueba el algoritmo cuántico de Grover."""
        # Configurar el sistema para usar Grover
        self.alert_system.quantum_simulator_type = "grover"
        
        # Crear coordenadas de prueba con suficientes puntos para calcular aceleraciones
        coords = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [5, 7, 9]])
        min_distance = 1.73  # Aproximadamente sqrt(3)
        
        # Calcular el riesgo con Grover
        risk = self.alert_system._apply_quantum_algorithm(coords, min_distance)
        
        # Verificar que el riesgo está en el rango correcto
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
    
    def test_publish_alert(self):
        """Prueba la publicación de alertas en Kafka."""
        # Crear una alerta de prueba
        alert = {
            "alert_id": "QA-20230101000000-SAT-001",
            "satellite_id": "SAT-001",
            "collision_probability": 0.75,
            "alert_level": "ALTO"
        }
        
        # Publicar la alerta
        self.alert_system.publish_alert(alert)
        
        # Verificar que se llamó al productor con los parámetros correctos
        self.producer_mock.produce.assert_called_once()
        args, kwargs = self.producer_mock.produce.call_args
        
        # Verificar que se usó el tópico correcto
        self.assertEqual(args[0], 'collision-alerts')
        
        # Verificar que la clave es el ID del satélite
        self.assertEqual(kwargs['key'], 'SAT-001')
        
        # Verificar que el valor es un JSON válido con la información de la alerta
        alert_json = json.loads(kwargs['value'])
        self.assertEqual(alert_json["alert_id"], "QA-20230101000000-SAT-001")
        self.assertEqual(alert_json["collision_probability"], 0.75)

if __name__ == '__main__':
    unittest.main()