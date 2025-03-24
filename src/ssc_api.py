import json
import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from .config import logger

class SSCApi:
    """
    Cliente para la API del Space Science Center (SSC) que maneja solicitudes
    y respuestas en formato JSON con las peculiaridades requeridas por la API.
    """
    def __init__(self):
        self.logger = logging.getLogger("Orbix.SSCApi")
        from .config import SSC_API_URL, SSC_API_KEY
        self.BASE_URL = SSC_API_URL
        self.API_KEY = SSC_API_KEY
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Api-Key": self.API_KEY
        })
    
    def _format_time(self, dt: datetime) -> str:
        """
        Formatea una fecha y hora en el formato requerido por la API SSC.
        """
        return dt.strftime("%Y-%m-%dT%H:%M:%S.000+00:00")
    
    def create_data_request(self, 
                           satellites: List[str], 
                           start_time: datetime, 
                           end_time: datetime,
                           resolution_factor: int = 1) -> Dict[str, Any]:
        """
        Crea una solicitud de datos en el formato JSON requerido por la API SSC.
        
        Args:
            satellites: Lista de IDs de satélites
            start_time: Tiempo de inicio para los datos
            end_time: Tiempo de fin para los datos
            resolution_factor: Factor de resolución para los datos (1=máxima resolución)
            
        Returns:
            Diccionario con la solicitud formateada para la API SSC
        """
        # Crear especificaciones de satélites en el formato requerido
        satellite_specs = [
            [
                "gov.nasa.gsfc.sscweb.schema.SatelliteSpecification",
                {
                    "Id": sat_id,
                    "ResolutionFactor": resolution_factor
                }
            ] for sat_id in satellites
        ]
        
        # Construir la solicitud completa con el formato específico requerido
        request_data = [
            "gov.nasa.gsfc.sscweb.schema.DataRequest",
            {
                "Description": "Orbital data request from Orbix",
                "TimeInterval": [
                    "gov.nasa.gsfc.sscweb.schema.TimeInterval",
                    {
                        "Start": [
                            "javax.xml.datatype.XMLGregorianCalendar",
                            self._format_time(start_time)
                        ],
                        "End": [
                            "javax.xml.datatype.XMLGregorianCalendar",
                            self._format_time(end_time)
                        ]
                    }
                ],
                "Satellites": [
                    "java.util.ArrayList",
                    satellite_specs
                ],
                "OutputOptions": [
                    "gov.nasa.gsfc.sscweb.schema.OutputOptions",
                    {
                        "CoordinateSystem": "gse",
                        "AllLocationFilters": True,
                        "RegionFilterDistance": 0.0,
                        "MinMaxPoints": 2
                    }
                ]
            }
        ]
        
        return request_data
    
    def get_satellite_data(self, 
                          satellites: List[str], 
                          start_time: datetime, 
                          end_time: datetime,
                          resolution_factor: int = 1) -> Dict[str, Any]:
        """
        Obtiene datos de satélites de la API SSC.
        
        Args:
            satellites: Lista de IDs de satélites
            start_time: Tiempo de inicio para los datos
            end_time: Tiempo de fin para los datos
            resolution_factor: Factor de resolución para los datos (1=máxima resolución)
            
        Returns:
            Datos de satélites procesados
        """
        request_data = self.create_data_request(
            satellites, start_time, end_time, resolution_factor
        )
        
        try:
            response = self.session.post(
                f"{self.BASE_URL}/locations", 
                json=request_data
            )
            response.raise_for_status()
            return self._process_response(response.json())
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener datos de satélites: {str(e)}")
            return {"error": str(e)}
    
    def _process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa la respuesta de la API SSC y la convierte a un formato más
        utilizable para el sistema Orbix.
        
        Args:
            response_data: Datos de respuesta de la API SSC
            
        Returns:
            Datos procesados en formato compatible con Orbix
        """
        try:
            # Extraer los datos relevantes de la estructura JSON compleja
            # La estructura exacta dependerá de la respuesta de la API
            result = {}
            
            # Verificar si hay un error en la respuesta
            if "Exception" in str(response_data):
                error_msg = str(response_data)
                self.logger.error(f"Error en la respuesta de la API SSC: {error_msg}")
                return {"error": error_msg}
            
            # Extraer datos de satélites
            # Nota: Esta parte puede necesitar ajustes según la estructura exacta de la respuesta
            if isinstance(response_data, list) and len(response_data) > 1:
                data_container = response_data[1]
                if "Result" in data_container and isinstance(data_container["Result"], list):
                    result_data = data_container["Result"][1]
                    
                    # Procesar datos para cada satélite
                    satellites_data = {}
                    for sat_data in result_data:
                        if isinstance(sat_data, list) and len(sat_data) > 1:
                            sat_id = sat_data[1].get("Id")
                            if sat_id:
                                # Extraer coordenadas y tiempos
                                coordinates = sat_data[1].get("Coordinates", [])
                                times = sat_data[1].get("Time", [])
                                
                                # Convertir a formato utilizable
                                trajectory_data = []
                                for i in range(min(len(coordinates), len(times))):
                                    if i < len(coordinates) and i < len(times):
                                        point = {
                                            "time": times[i],
                                            "x": coordinates[i][0],
                                            "y": coordinates[i][1],
                                            "z": coordinates[i][2]
                                        }
                                        trajectory_data.append(point)
                                
                                satellites_data[sat_id] = trajectory_data
                    
                    result["satellites"] = satellites_data
            
            return result
        except Exception as e:
            self.logger.error(f"Error al procesar la respuesta: {str(e)}")
            return {"error": str(e)}
    
    def get_available_satellites(self) -> List[str]:
        """
        Obtiene la lista de satélites disponibles en la API SSC.
        
        Returns:
            Lista de IDs de satélites disponibles
        """
        try:
            response = self.session.get(f"{self.BASE_URL}/observatories")
            response.raise_for_status()
            
            # Procesar la respuesta para extraer los IDs de satélites
            # La estructura exacta dependerá de la respuesta de la API
            satellites = []
            response_data = response.json()
            
            # Extraer IDs de satélites de la respuesta
            # Nota: Esta parte puede necesitar ajustes según la estructura exacta de la respuesta
            if isinstance(response_data, list) and len(response_data) > 1:
                data_container = response_data[1]
                if "Observatory" in data_container and isinstance(data_container["Observatory"], list):
                    for sat in data_container["Observatory"][1]:
                        if isinstance(sat, list) and len(sat) > 1:
                            sat_id = sat[1].get("Id")
                            if sat_id:
                                satellites.append(sat_id)
            
            return satellites
        except requests.RequestException as e:
            self.logger.error(f"Error al obtener satélites disponibles: {str(e)}")
            return []