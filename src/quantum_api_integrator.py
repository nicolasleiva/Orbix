import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from .quantum_alerts import QuantumCollisionAlertSystem
from .quantum_trajectory import QuantumTrajectoryModel
from .ssc_api import SSCApi
from .spacex_api import SpaceXApi
from .space_track_api import SpaceTrackApi
from .noaa_api import NOAAApi
from .config import logger

class QuantumApiIntegrator:
    """
    Integrador que conecta los módulos cuánticos (alertas y trayectorias) con las APIs
    externas para obtener datos orbitales reales y procesarlos con algoritmos cuánticos.
    
    Esta clase actúa como puente entre las fuentes de datos (SSC, SpaceX, Space-Track, NOAA)
    y los algoritmos cuánticos, permitiendo un flujo de datos coherente y optimizado.
    """
    
    def __init__(self):
        """
        Inicializa el integrador con las instancias de los módulos cuánticos y las APIs.
        """
        self.logger = logging.getLogger("Orbix.QuantumApiIntegrator")
        
        # Inicializar módulos cuánticos
        self.alert_system = QuantumCollisionAlertSystem()
        self.trajectory_model = QuantumTrajectoryModel()
        
        # Inicializar APIs
        self.ssc_api = SSCApi()
        self.spacex_api = SpaceXApi()
        self.space_track_api = SpaceTrackApi()
        self.noaa_api = NOAAApi()
        
        self.logger.info("Integrador de APIs cuánticas inicializado correctamente")
    
    def get_satellite_data(self, satellite_id: str, start_time: datetime, 
                          end_time: datetime) -> Dict[str, Any]:
        """
        Obtiene datos de un satélite específico combinando información de múltiples APIs.
        
        Args:
            satellite_id: Identificador del satélite (NORAD ID o nombre)
            start_time: Tiempo de inicio para los datos
            end_time: Tiempo de fin para los datos
            
        Returns:
            Dict con datos combinados del satélite de múltiples fuentes
        """
        result = {}
        
        try:
            # Intentar obtener datos de Space-Track (datos TLE)
            try:
                norad_id = int(satellite_id) if satellite_id.isdigit() else None
                if norad_id:
                    tle_data = self.space_track_api.get_latest_tle(norad_id)
                    if "error" not in tle_data:
                        result["tle_data"] = tle_data
            except Exception as e:
                self.logger.warning(f"Error al obtener datos TLE de Space-Track: {str(e)}")
            
            # Obtener datos de posición de SSC
            try:
                ssc_data = self.ssc_api.get_satellite_data(
                    satellites=[satellite_id],
                    start_time=start_time,
                    end_time=end_time
                )
                if "error" not in ssc_data:
                    result["position_data"] = ssc_data
            except Exception as e:
                self.logger.warning(f"Error al obtener datos de posición de SSC: {str(e)}")
            
            # Obtener datos de clima espacial de NOAA
            try:
                space_weather = self.noaa_api.get_solar_wind_data()
                if "error" not in space_weather:
                    result["space_weather"] = space_weather
            except Exception as e:
                self.logger.warning(f"Error al obtener datos de clima espacial: {str(e)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error al obtener datos del satélite {satellite_id}: {str(e)}")
            return {"error": str(e)}
    
    def convert_api_data_to_trajectory(self, api_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Convierte datos de las APIs a un DataFrame con formato de trayectoria
        compatible con los módulos cuánticos.
        
        Args:
            api_data: Datos obtenidos de las APIs
            
        Returns:
            DataFrame con la trayectoria en formato [x, y, z, timestamp]
        """
        try:
            # Verificar si tenemos datos de posición
            if "position_data" not in api_data or "data" not in api_data["position_data"]:
                raise ValueError("No hay datos de posición disponibles")
            
            position_data = api_data["position_data"]["data"]
            
            # Extraer coordenadas y timestamps
            coordinates = []
            timestamps = []
            
            for point in position_data:
                if "coordinates" in point and "time" in point:
                    coords = point["coordinates"]
                    if "x" in coords and "y" in coords and "z" in coords:
                        coordinates.append([coords["x"], coords["y"], coords["z"]])
                        timestamps.append(point["time"])
            
            if not coordinates:
                raise ValueError("No se pudieron extraer coordenadas válidas")
            
            # Crear DataFrame
            trajectory_df = pd.DataFrame(coordinates, columns=["x", "y", "z"])
            trajectory_df["timestamp"] = timestamps
            
            return trajectory_df
            
        except Exception as e:
            self.logger.error(f"Error al convertir datos a trayectoria: {str(e)}")
            # Devolver DataFrame vacío en caso de error
            return pd.DataFrame(columns=["x", "y", "z", "timestamp"])
    
    def predict_trajectory(self, satellite_id: str, start_time: datetime, 
                          end_time: datetime, prediction_hours: int = 24) -> pd.DataFrame:
        """
        Predice la trayectoria futura de un satélite utilizando el modelo cuántico.
        
        Args:
            satellite_id: Identificador del satélite
            start_time: Tiempo de inicio para los datos históricos
            end_time: Tiempo de fin para los datos históricos
            prediction_hours: Número de horas a predecir en el futuro
            
        Returns:
            DataFrame con la trayectoria predicha
        """
        try:
            # Obtener datos históricos
            satellite_data = self.get_satellite_data(satellite_id, start_time, end_time)
            
            # Convertir a formato de trayectoria
            historical_trajectory = self.convert_api_data_to_trajectory(satellite_data)
            
            if historical_trajectory.empty:
                raise ValueError("No hay suficientes datos históricos para la predicción")
            
            # Preparar datos para el modelo cuántico
            # Convertir DataFrame a tensor para el modelo
            input_data = historical_trajectory[["x", "y", "z"]].to_numpy()
            input_tensor = np.expand_dims(input_data, axis=0)  # Añadir dimensión de batch
            
            # Convertir a tensor de TensorFlow
            import tensorflow as tf
            tf_input = tf.convert_to_tensor(input_tensor, dtype=tf.float32)
            
            # Realizar predicción con el modelo cuántico
            predicted_coords = self.trajectory_model(tf_input).numpy()[0]  # Eliminar dimensión de batch
            
            # Crear DataFrame con la predicción
            timestamps = []
            last_timestamp = historical_trajectory["timestamp"].iloc[-1] if "timestamp" in historical_trajectory.columns else datetime.now()
            
            if isinstance(last_timestamp, str):
                last_timestamp = datetime.fromisoformat(last_timestamp.replace('Z', '+00:00'))
            
            # Generar timestamps futuros
            for i in range(len(predicted_coords)):
                future_time = last_timestamp + timedelta(hours=i * prediction_hours / len(predicted_coords))
                timestamps.append(future_time)
            
            # Crear DataFrame con la predicción
            prediction_df = pd.DataFrame(predicted_coords, columns=["x", "y", "z"])
            prediction_df["timestamp"] = timestamps
            
            return prediction_df
            
        except Exception as e:
            self.logger.error(f"Error al predecir trayectoria: {str(e)}")
            return pd.DataFrame(columns=["x", "y", "z", "timestamp"])
    
    def generate_collision_alert(self, satellite_id: str, other_object_id: str = None,
                               prediction_hours: int = 72) -> Dict[str, Any]:
        """
        Genera una alerta de colisión para un satélite utilizando el sistema de alertas cuánticas.
        
        Args:
            satellite_id: Identificador del satélite principal
            other_object_id: Identificador del otro objeto (opcional)
            prediction_hours: Número de horas a predecir para evaluar colisión
            
        Returns:
            Dict con la información de la alerta
        """
        try:
            # Definir ventana de tiempo para datos históricos (últimas 24 horas)
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=24)
            
            # Predecir trayectoria futura
            predicted_trajectory = self.predict_trajectory(
                satellite_id=satellite_id,
                start_time=start_time,
                end_time=end_time,
                prediction_hours=prediction_hours
            )
            
            if predicted_trajectory.empty:
                raise ValueError("No se pudo generar una predicción de trayectoria válida")
            
            # Si se especificó otro objeto, obtener también su trayectoria
            other_trajectory = None
            if other_object_id:
                other_trajectory = self.predict_trajectory(
                    satellite_id=other_object_id,
                    start_time=start_time,
                    end_time=end_time,
                    prediction_hours=prediction_hours
                )
            
            # Obtener datos de clima espacial para metadatos adicionales
            space_weather = self.noaa_api.get_solar_wind_data()
            
            # Preparar metadatos adicionales
            additional_metadata = {
                "prediction_hours": prediction_hours,
                "data_sources": ["SSC", "SpaceTrack", "NOAA"],
                "space_weather": space_weather.get("data", [])
            }
            
            # Generar alerta con el sistema cuántico
            alert = self.alert_system.generate_alert(
                satellite_id=satellite_id,
                trajectory=predicted_trajectory,
                other_object_id=other_object_id,
                additional_metadata=additional_metadata
            )
            
            # Publicar alerta
            self.alert_system.publish_alert(alert)
            
            return alert
            
        except Exception as e:
            self.logger.error(f"Error al generar alerta de colisión: {str(e)}")
            return {"error": str(e)}
    
    def analyze_multiple_satellites(self, satellite_ids: List[str], 
                                  prediction_hours: int = 72) -> List[Dict[str, Any]]:
        """
        Analiza múltiples satélites para detectar posibles colisiones entre ellos.
        
        Args:
            satellite_ids: Lista de identificadores de satélites
            prediction_hours: Número de horas a predecir para evaluar colisiones
            
        Returns:
            Lista de alertas para las posibles colisiones detectadas
        """
        alerts = []
        
        try:
            # Predecir trayectorias para todos los satélites
            trajectories = {}
            for sat_id in satellite_ids:
                trajectory = self.predict_trajectory(
                    satellite_id=sat_id,
                    start_time=datetime.now() - timedelta(hours=24),
                    end_time=datetime.now(),
                    prediction_hours=prediction_hours
                )
                if not trajectory.empty:
                    trajectories[sat_id] = trajectory
            
            # Comparar cada par de satélites
            for i, sat1 in enumerate(satellite_ids):
                if sat1 not in trajectories:
                    continue
                    
                for j in range(i+1, len(satellite_ids)):
                    sat2 = satellite_ids[j]
                    if sat2 not in trajectories:
                        continue
                    
                    # Generar alerta para este par
                    alert = self.alert_system.generate_alert(
                        satellite_id=sat1,
                        trajectory=trajectories[sat1],
                        other_object_id=sat2,
                        additional_metadata={
                            "comparison_type": "satellite-satellite",
                            "prediction_hours": prediction_hours
                        }
                    )
                    
                    # Solo añadir alertas con probabilidad significativa
                    if alert["collision_probability"] > 0.1:
                        alerts.append(alert)
                        # Publicar alerta
                        self.alert_system.publish_alert(alert)
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Error al analizar múltiples satélites: {str(e)}")
            return [{"error": str(e)}]