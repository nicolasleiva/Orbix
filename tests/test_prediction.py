import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch, MagicMock
from src.prediction import predict_trajectory, load_model
from src.transformer_model import TransformerTrajectoryModel

class TestPrediction(unittest.TestCase):
    """Pruebas unitarias para el módulo de predicción."""
    
    @patch('src.prediction.global_model')
    def test_predict_trajectory_valid_input(self, mock_model):
        """Prueba que la predicción funcione correctamente con datos válidos."""
        # Configurar el mock
        mock_output = tf.constant([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
        mock_model.return_value = mock_output
        
        # Datos de entrada
        tle_data = {
            "satellite_id": "12345",
            "data": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        
        # Ejecutar la función
        result = predict_trajectory(tle_data)
        
        # Verificar el resultado
        self.assertEqual(result["satellite_id"], "12345")
        self.assertEqual(len(result["trajectory"]), 1)
        self.assertEqual(len(result["trajectory"][0]), 2)
        self.assertEqual(len(result["trajectory"][0][0]), 3)
    
    def test_predict_trajectory_empty_data(self):
        """Prueba que la predicción maneje correctamente datos vacíos."""
        # Datos de entrada vacíos
        tle_data = {"satellite_id": "12345", "data": []}
        
        # Ejecutar la función
        result = predict_trajectory(tle_data)
        
        # Verificar que se devuelve un error
        self.assertIn("error", result)
    
    @patch('src.prediction.TransformerTrajectoryModel')
    @patch('src.prediction.global_model')
    def test_load_model(self, mock_global_model, mock_transformer):
        """Prueba que el modelo se cargue correctamente."""
        # Configurar los mocks
        mock_model_instance = MagicMock()
        mock_transformer.return_value = mock_model_instance
        
        # Ejecutar la función
        load_model()
        
        # Verificar que se llamó a load_weights
        mock_model_instance.load_weights.assert_called_once()

if __name__ == '__main__':
    unittest.main()