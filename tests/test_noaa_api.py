import unittest
from unittest.mock import patch, MagicMock
import json
import os
import sys

# Agregar el directorio src al path para poder importar los módulos
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.noaa_api import NOAAApi


class TestNOAAApi(unittest.TestCase):
    """Pruebas unitarias para la clase NOAAApi"""

    def setUp(self):
        """Configuración inicial para las pruebas"""
        self.noaa_api = NOAAApi()
        
        # Datos de ejemplo para simular respuestas de la API
        self.sample_solar_wind_data = [
            {"time_tag": "2023-01-01T00:00:00Z", "bx": 1.2, "by": -0.5, "bz": 0.8, "bt": 1.5, "density": 5.2, "speed": 450.3},
            {"time_tag": "2023-01-01T01:00:00Z", "bx": 1.3, "by": -0.6, "bz": 0.7, "bt": 1.6, "density": 5.3, "speed": 451.2}
        ]
        
        self.sample_geomag_indices = [
            {"time_tag": "2023-01-01T00:00:00Z", "kp_index": 3, "dst_index": -15, "ap_index": 12},
            {"time_tag": "2023-01-01T03:00:00Z", "kp_index": 4, "dst_index": -20, "ap_index": 15}
        ]
        
        self.sample_solar_flare_data = [
            {"time_tag": "2023-01-01T00:00:00Z", "class": "C1.2", "intensity": 1.2e-6, "integrated_flux": 1.5e-4},
            {"time_tag": "2023-01-01T12:00:00Z", "class": "M2.3", "intensity": 2.3e-5, "integrated_flux": 3.2e-3}
        ]
        
        self.sample_proton_flux_data = [
            {"time_tag": "2023-01-01T00:00:00Z", "p1": 10.2, "p5": 5.1, "p10": 2.3, "p30": 0.5, "p50": 0.2, "p100": 0.1},
            {"time_tag": "2023-01-01T01:00:00Z", "p1": 11.3, "p5": 5.5, "p10": 2.5, "p30": 0.6, "p50": 0.3, "p100": 0.1}
        ]
        
        self.sample_aurora_forecast = [
            {"time_tag": "2023-01-01T00:00:00Z", "latitude": 60.0, "longitude": -100.0, "probability": 0.75},
            {"time_tag": "2023-01-01T00:00:00Z", "latitude": 65.0, "longitude": -105.0, "probability": 0.85}
        ]
        
        self.sample_space_weather_alerts = [
            {"issue_time": "2023-01-01T00:00:00Z", "message_code": "ALTXMF", "message": "X-class Solar Flare", "severity": "WARNING"},
            {"issue_time": "2023-01-01T12:00:00Z", "message_code": "WATA20", "message": "Geomagnetic Storm Watch", "severity": "WATCH"}
        ]

    @patch('requests.Session.get')
    def test_get_solar_wind_data(self, mock_get):
        """Prueba para obtener datos del viento solar"""
        # Configurar el mock para simular la respuesta de la API
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_solar_wind_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Llamar al método y verificar el resultado
        result = self.noaa_api.get_solar_wind_data()
        self.assertEqual(result, {"data": self.sample_solar_wind_data})
        mock_get.assert_called_once_with(f"{self.noaa_api.BASE_URL}/solar-wind/mag-1-day.json")

    @patch('requests.Session.get')
    def test_get_geomagnetic_indices(self, mock_get):
        """Prueba para obtener índices geomagnéticos"""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_geomag_indices
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.noaa_api.get_geomagnetic_indices()
        self.assertEqual(result, {"data": self.sample_geomag_indices})
        mock_get.assert_called_once_with(f"{self.noaa_api.BASE_URL}/geospace/geomag-indices-1-day.json")

    @patch('requests.Session.get')
    def test_get_solar_flare_data(self, mock_get):
        """Prueba para obtener datos de llamaradas solares"""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_solar_flare_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.noaa_api.get_solar_flare_data(days=1)
        self.assertEqual(result, {"data": self.sample_solar_flare_data})
        mock_get.assert_called_once_with(f"{self.noaa_api.BASE_URL}/goes/primary-xrays-1-day.json")

    @patch('requests.Session.get')
    def test_get_proton_flux_data(self, mock_get):
        """Prueba para obtener datos de flujo de protones"""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_proton_flux_data
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.noaa_api.get_proton_flux_data()
        self.assertEqual(result, {"data": self.sample_proton_flux_data})
        mock_get.assert_called_once_with(f"{self.noaa_api.BASE_URL}/goes/proton-fluences-1-day.json")

    @patch('requests.Session.get')
    def test_get_aurora_forecast(self, mock_get):
        """Prueba para obtener pronóstico de auroras"""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_aurora_forecast
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.noaa_api.get_aurora_forecast()
        self.assertEqual(result, {"data": self.sample_aurora_forecast})
        mock_get.assert_called_once_with(f"{self.noaa_api.BASE_URL}/ovation/aurora-forecast-map.json")

    @patch('requests.Session.get')
    def test_get_space_weather_alerts(self, mock_get):
        """Prueba para obtener alertas de clima espacial"""
        mock_response = MagicMock()
        mock_response.json.return_value = self.sample_space_weather_alerts
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = self.noaa_api.get_space_weather_alerts()
        self.assertEqual(result, {"data": self.sample_space_weather_alerts})
        mock_get.assert_called_once_with(f"{self.noaa_api.BASE_URL}/alerts.json")

    @patch('requests.Session.get')
    def test_error_handling(self, mock_get):
        """Prueba para verificar el manejo de errores"""
        mock_get.side_effect = requests.RequestException("Error de conexión")
        
        result = self.noaa_api.get_solar_wind_data()
        self.assertEqual(result, {"error": "Error de conexión"})


if __name__ == '__main__':
    unittest.main()