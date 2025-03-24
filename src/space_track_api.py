import json
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from .config import logger

class SpaceTrackApi:
    """
    Cliente para la API de Space-Track que maneja solicitudes
    y respuestas en formato JSON para obtener información sobre objetos espaciales,
    TLEs, y datos de conjunciones.
    """
    def __init__(self):
        self.logger = logging.getLogger("Orbix.SpaceTrackApi")
        from .config import SPACE_TRACK_USER, SPACE_TRACK_PASS
        self.BASE_URL = "https://www.space-track.org"
        self.AUTH_URL = f"{self.BASE_URL}/ajaxauth/login"
        self.username = SPACE_TRACK_USER
        self.password = SPACE_TRACK_PASS
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _login(self) -> bool:
        """
        Inicia sesión en la API de Space-Track.
        
        Returns:
            True si el login fue exitoso, False en caso contrario
        """
        try:
            payload = {
                "identity": self.username,
                "password": self.password
            }
            response = self.session.post(self.AUTH_URL, data=payload)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            self.logger.error(f"Error al iniciar sesión en Space-Track: {str(e)}")
            return False
    
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
    
    def get_satellite_catalog(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtiene el catálogo de satélites de Space-Track.
        
        Args:
            limit: Número máximo de satélites a obtener
            
        Returns:
            Información sobre satélites
        """
        if not self._login():
            return {"error": "Error de autenticación"}
        
        try:
            url = f"{self.BASE_URL}/basicspacedata/query/class/satcat"
            params = {"format": "json"}
            if limit is not None:
                params["limit"] = limit
                
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener catálogo de satélites: {str(e)}")
            return {"error": str(e)}
    
    def get_latest_tle(self, norad_id: int) -> Dict[str, Any]:
        """
        Obtiene el TLE más reciente para un satélite específico.
        
        Args:
            norad_id: ID NORAD del satélite
            
        Returns:
            TLE más reciente del satélite
        """
        if not self._login():
            return {"error": "Error de autenticación"}
        
        try:
            url = f"{self.BASE_URL}/basicspacedata/query/class/tle_latest/NORAD_CAT_ID/{norad_id}/orderby/EPOCH desc/limit/1"
            params = {"format": "json"}
                
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener TLE para el satélite {norad_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_conjunction_data(self, days_from_now: int = 7) -> Dict[str, Any]:
        """
        Obtiene datos de conjunciones (posibles colisiones) para los próximos días.
        
        Args:
            days_from_now: Número de días hacia adelante para obtener datos
            
        Returns:
            Datos de conjunciones
        """
        if not self._login():
            return {"error": "Error de autenticación"}
        
        try:
            now = datetime.utcnow()
            future = now + timedelta(days=days_from_now)
            start_date = now.strftime("%Y-%m-%d")
            end_date = future.strftime("%Y-%m-%d")
            
            url = f"{self.BASE_URL}/basicspacedata/query/class/cdm_public/CREATION_DATE/>/{start_date}/CREATION_DATE/</{end_date}"
            params = {"format": "json"}
                
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener datos de conjunciones: {str(e)}")
            return {"error": str(e)}
    
    def get_decay_data(self, days_from_now: int = 30) -> Dict[str, Any]:
        """
        Obtiene datos de decaimiento orbital para los próximos días.
        
        Args:
            days_from_now: Número de días hacia adelante para obtener datos
            
        Returns:
            Datos de decaimiento orbital
        """
        if not self._login():
            return {"error": "Error de autenticación"}
        
        try:
            now = datetime.utcnow()
            future = now + timedelta(days=days_from_now)
            start_date = now.strftime("%Y-%m-%d")
            end_date = future.strftime("%Y-%m-%d")
            
            url = f"{self.BASE_URL}/basicspacedata/query/class/decay/DECAY_DATE/>/{start_date}/DECAY_DATE/</{end_date}"
            params = {"format": "json"}
                
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener datos de decaimiento orbital: {str(e)}")
            return {"error": str(e)}