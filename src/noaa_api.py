import json
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .config import logger

class NOAAApi:
    """
    Cliente para la API de NOAA (National Oceanic and Atmospheric Administration) que maneja solicitudes
    y respuestas en formato JSON para obtener datos meteorológicos espaciales, incluyendo viento solar,
    índices geomagnéticos, llamaradas solares y alertas.
    """
    def __init__(self):
        self.logger = logging.getLogger("Orbix.NOAAApi")
        from .config import NOAA_TOKEN
        self.BASE_URL = "https://services.swpc.noaa.gov/json"
        self.token = NOAA_TOKEN
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.token}"
        })
    
    def _process_response(self, data: Any) -> Dict[str, Any]:
        """
        Procesa la respuesta de la API y la convierte en un formato estándar.
        
        Args:
            data: Datos de respuesta de la API
            
        Returns:
            Datos procesados en formato estándar
        """
        if isinstance(data, list):
            return {"data": data}
        return {"data": [data]} if data else {"data": []}
    
    def get_solar_wind_data(self) -> Dict[str, Any]:
        """
        Obtiene datos actuales del viento solar.
        
        Returns:
            Datos del viento solar
        """
        try:
            url = f"{self.BASE_URL}/solar-wind/mag-1-day.json"
            response = self.session.get(url)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener datos del viento solar: {str(e)}")
            return {"error": str(e)}
    
    def get_geomagnetic_indices(self) -> Dict[str, Any]:
        """
        Obtiene índices geomagnéticos actuales.
        
        Returns:
            Índices geomagnéticos
        """
        try:
            url = f"{self.BASE_URL}/geospace/geomag-indices-1-day.json"
            response = self.session.get(url)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener índices geomagnéticos: {str(e)}")
            return {"error": str(e)}
    
    def get_solar_flare_data(self, days: int = 1) -> Dict[str, Any]:
        """
        Obtiene datos de llamaradas solares para un período específico.
        
        Args:
            days: Número de días hacia atrás para obtener datos
            
        Returns:
            Datos de llamaradas solares
        """
        try:
            url = f"{self.BASE_URL}/goes/primary-xrays-{days}-day.json"
            response = self.session.get(url)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener datos de llamaradas solares: {str(e)}")
            return {"error": str(e)}
    
    def get_proton_flux_data(self) -> Dict[str, Any]:
        """
        Obtiene datos actuales de flujo de protones.
        
        Returns:
            Datos de flujo de protones
        """
        try:
            url = f"{self.BASE_URL}/goes/proton-fluences-1-day.json"
            response = self.session.get(url)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener datos de flujo de protones: {str(e)}")
            return {"error": str(e)}
    
    def get_aurora_forecast(self) -> Dict[str, Any]:
        """
        Obtiene pronóstico actual de auroras.
        
        Returns:
            Pronóstico de auroras
        """
        try:
            url = f"{self.BASE_URL}/ovation/aurora-forecast-map.json"
            response = self.session.get(url)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener pronóstico de auroras: {str(e)}")
            return {"error": str(e)}
    
    def get_space_weather_alerts(self) -> Dict[str, Any]:
        """
        Obtiene alertas actuales de clima espacial.
        
        Returns:
            Alertas de clima espacial
        """
        try:
            url = f"{self.BASE_URL}/alerts.json"
            response = self.session.get(url)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener alertas de clima espacial: {str(e)}")
            return {"error": str(e)}