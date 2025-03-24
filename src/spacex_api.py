import json
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from .config import logger

class SpaceXApi:
    """
    Cliente para la API de SpaceX que maneja solicitudes
    y respuestas en formato JSON para obtener información sobre lanzamientos,
    cohetes, cápsulas y misiones de SpaceX.
    """
    def __init__(self):
        self.logger = logging.getLogger("Orbix.SpaceXApi")
        from .config import SPACEX_API_URL
        self.BASE_URL = SPACEX_API_URL
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def get_launches(self, limit: Optional[int] = None, offset: Optional[int] = None) -> Dict[str, Any]:
        """
        Obtiene información sobre lanzamientos de SpaceX.
        
        Args:
            limit: Número máximo de lanzamientos a obtener
            offset: Número de lanzamientos a saltar
            
        Returns:
            Información sobre lanzamientos de SpaceX
        """
        params = {}
        if limit is not None:
            params['limit'] = limit
        if offset is not None:
            params['offset'] = offset
            
        try:
            response = self.session.get(f"{self.BASE_URL}/launches", params=params)
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener lanzamientos: {str(e)}")
            return {"error": str(e)}
    
    def get_launch(self, flight_number: int) -> Dict[str, Any]:
        """
        Obtiene información sobre un lanzamiento específico de SpaceX.
        
        Args:
            flight_number: Número de vuelo del lanzamiento
            
        Returns:
            Información sobre el lanzamiento
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/launches/{flight_number}")
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener lanzamiento {flight_number}: {str(e)}")
            return {"error": str(e)}
    
    def get_rockets(self) -> Dict[str, Any]:
        """
        Obtiene información sobre cohetes de SpaceX.
        
        Returns:
            Información sobre cohetes de SpaceX
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/rockets")
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener cohetes: {str(e)}")
            return {"error": str(e)}
    
    def get_rocket(self, rocket_id: str) -> Dict[str, Any]:
        """
        Obtiene información sobre un cohete específico de SpaceX.
        
        Args:
            rocket_id: ID del cohete
            
        Returns:
            Información sobre el cohete
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/rockets/{rocket_id}")
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener cohete {rocket_id}: {str(e)}")
            return {"error": str(e)}
    
    def get_capsules(self) -> Dict[str, Any]:
        """
        Obtiene información sobre cápsulas de SpaceX.
        
        Returns:
            Información sobre cápsulas de SpaceX
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/capsules")
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener cápsulas: {str(e)}")
            return {"error": str(e)}
    
    def get_capsule(self, capsule_serial: str) -> Dict[str, Any]:
        """
        Obtiene información sobre una cápsula específica de SpaceX.
        
        Args:
            capsule_serial: Número de serie de la cápsula
            
        Returns:
            Información sobre la cápsula
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/capsules/{capsule_serial}")
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener cápsula {capsule_serial}: {str(e)}")
            return {"error": str(e)}
    
    def _process_response(self, response_data: Any) -> Dict[str, Any]:
        """
        Procesa la respuesta de la API de SpaceX y la convierte a un formato
        utilizable para el sistema Orbix.
        
        Args:
            response_data: Datos de respuesta de la API de SpaceX
            
        Returns:
            Datos procesados en formato compatible con Orbix
        """
        try:
            # La API de SpaceX devuelve datos en un formato más simple que la API SSC,
            # por lo que en la mayoría de los casos solo necesitamos devolver los datos tal como están
            if isinstance(response_data, list):
                return {"data": response_data}
            elif isinstance(response_data, dict):
                return response_data
            else:
                return {"data": response_data}
        except Exception as e:
            self.logger.error(f"Error al procesar la respuesta: {str(e)}")
            return {"error": str(e)}