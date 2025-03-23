import unittest
from unittest.mock import patch, MagicMock
from src.ml_pipeline import run_tfx_pipeline

class TestMLPipeline(unittest.TestCase):
    """Pruebas unitarias para el pipeline de Machine Learning."""
    
    @patch('src.ml_pipeline.InteractiveContext')
    @patch('src.ml_pipeline.CsvExampleGen')
    @patch('src.ml_pipeline.StatisticsGen')
    @patch('src.ml_pipeline.SchemaGen')
    @patch('src.ml_pipeline.Transform')
    @patch('src.ml_pipeline.Trainer')
    @patch('src.ml_pipeline.Pusher')
    def test_run_tfx_pipeline(self, mock_pusher, mock_trainer, mock_transform, 
                             mock_schema_gen, mock_statistics_gen, 
                             mock_example_gen, mock_context):
        """Prueba que el pipeline TFX se ejecute correctamente."""
        # Configurar los mocks
        mock_context_instance = MagicMock()
        mock_context.return_value = mock_context_instance
        
        # Configurar los mocks para los componentes TFX
        mock_example_gen_instance = MagicMock()
        mock_example_gen.return_value = mock_example_gen_instance
        mock_example_gen_instance.outputs = {'examples': MagicMock()}
        
        mock_statistics_gen_instance = MagicMock()
        mock_statistics_gen.return_value = mock_statistics_gen_instance
        mock_statistics_gen_instance.outputs = {'statistics': MagicMock()}
        
        mock_schema_gen_instance = MagicMock()
        mock_schema_gen.return_value = mock_schema_gen_instance
        mock_schema_gen_instance.outputs = {'schema': MagicMock()}
        
        mock_transform_instance = MagicMock()
        mock_transform.return_value = mock_transform_instance
        mock_transform_instance.outputs = {
            'transformed_examples': MagicMock(),
            'transform_graph': MagicMock()
        }
        
        mock_trainer_instance = MagicMock()
        mock_trainer.return_value = mock_trainer_instance
        mock_trainer_instance.outputs = {'model': MagicMock()}
        
        # Ejecutar la función
        run_tfx_pipeline()
        
        # Verificar que se crearon todos los componentes
        mock_example_gen.assert_called_once_with(input_base='data/')
        mock_statistics_gen.assert_called_once()
        mock_schema_gen.assert_called_once()
        mock_transform.assert_called_once()
        mock_trainer.assert_called_once()
        mock_pusher.assert_called_once()
        
        # Verificar que se ejecutó el pipeline
        mock_context_instance.run.assert_called_once()

if __name__ == '__main__':
    unittest.main()