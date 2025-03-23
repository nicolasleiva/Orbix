import unittest
import json
from unittest.mock import patch, MagicMock, call
from src.realtime_pipeline import RealTimeDataPipeline

class TestRealTimePipeline(unittest.TestCase):
    """Pruebas unitarias para el pipeline de datos en tiempo real."""
    
    @patch('src.realtime_pipeline.Producer')
    @patch('src.realtime_pipeline.Consumer')
    def setUp(self, mock_consumer, mock_producer):
        """Configuración inicial para las pruebas."""
        # Configurar los mocks
        self.mock_producer_instance = MagicMock()
        self.mock_consumer_instance = MagicMock()
        mock_producer.return_value = self.mock_producer_instance
        mock_consumer.return_value = self.mock_consumer_instance
        
        # Crear la instancia del pipeline
        self.pipeline = RealTimeDataPipeline()
        
        # Datos de prueba
        self.tle_data = {
            "satellite_id": "12345",
            "data": [0.1, 0.2, 0.3, 0.4, 0.5]
        }
        self.prediction_data = {
            "satellite_id": "12345",
            "trajectory": [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]
        }
    
    def test_init(self):
        """Prueba que la inicialización configure correctamente Producer y Consumer."""
        # Verificar que se crearon las instancias de Producer y Consumer
        self.assertIsNotNone(self.pipeline.producer)
        self.assertIsNotNone(self.pipeline.consumer)
    
    def test_stream_tle_data(self):
        """Prueba que el generador de datos TLE funcione correctamente."""
        # Configurar el mock para simular mensajes de Kafka
        mock_message = MagicMock()
        mock_message.error.return_value = None
        mock_message.value.return_value = json.dumps(self.tle_data).encode('utf-8')
        
        # Configurar el comportamiento del consumer.poll
        self.mock_consumer_instance.poll.side_effect = [mock_message, None]
        
        # Usar el generador
        generator = self.pipeline.stream_tle_data()
        result = next(generator)
        
        # Verificar que se llamó a subscribe con el tópico correcto
        self.mock_consumer_instance.subscribe.assert_called_once_with(['tle-updates'])
        
        # Verificar que se llamó a poll
        self.mock_consumer_instance.poll.assert_called_with(1.0)
        
        # Verificar que el resultado es el esperado
        self.assertEqual(result, self.tle_data)
    
    def test_stream_tle_data_error(self):
        """Prueba que el generador maneje correctamente los errores de Kafka."""
        # Configurar el mock para simular un error de Kafka
        mock_message = MagicMock()
        mock_message.error.return_value = "Error de conexión"
        
        # Configurar otro mensaje válido después del error
        mock_message_valid = MagicMock()
        mock_message_valid.error.return_value = None
        mock_message_valid.value.return_value = json.dumps(self.tle_data).encode('utf-8')
        
        # Configurar el comportamiento del consumer.poll
        self.mock_consumer_instance.poll.side_effect = [mock_message, mock_message_valid, None]
        
        # Usar el generador
        generator = self.pipeline.stream_tle_data()
        result = next(generator)
        
        # Verificar que se llamó a poll dos veces (una para el error, otra para el mensaje válido)
        self.assertEqual(self.mock_consumer_instance.poll.call_count, 2)
        
        # Verificar que el resultado es el esperado (del segundo mensaje)
        self.assertEqual(result, self.tle_data)
    
    def test_publish_prediction(self):
        """Prueba que la publicación de predicciones funcione correctamente."""
        # Llamar al método
        self.pipeline.publish_prediction(self.prediction_data)
        
        # Verificar que se llamó a produce con los parámetros correctos
        self.mock_producer_instance.produce.assert_called_once_with(
            'trajectory-predictions',
            key=self.prediction_data['satellite_id'],
            value=json.dumps(self.prediction_data)
        )
        
        # Verificar que se llamó a flush
        self.mock_producer_instance.flush.assert_called_once()

if __name__ == '__main__':
    unittest.main()