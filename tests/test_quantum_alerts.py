import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.quantum_alerts import QuantumCollisionAlertSystem

class TestQuantumAlerts(unittest.TestCase):
    """Pruebas unitarias para el sistema de alertas de colisión cuántica."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
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

if __name__ == '__main__':
    unittest.main()