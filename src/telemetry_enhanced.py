import tensorflow as tf
import time
import logging
from prometheus_client import Gauge, push_to_gateway

logger = logging.getLogger("Orbix")

class TelemetryMonitor:
    """
    Monitorea métricas en tiempo real de la predicción y optimización orbital.
    
    Esta clase proporciona funcionalidades para recopilar, actualizar y exportar
    métricas relacionadas con la predicción de trayectorias orbitales y la
    optimización de rutas para satélites.
    
    Attributes:
        metrics (dict): Diccionario de métricas de TensorFlow para seguimiento interno.
        prometheus_metrics (dict): Diccionario de métricas de Prometheus para exportación.
        last_update_time (float): Timestamp de la última actualización de métricas.
    """
    
    def __init__(self):
        """
        Inicializa el monitor de telemetría con métricas específicas para navegación orbital.
        """
        # Métricas internas de TensorFlow
        self.metrics = {
            'position_error': tf.keras.metrics.MeanSquaredError(),
            'velocity_error': tf.keras.metrics.MeanAbsoluteError(),
            'collision_probability': tf.keras.metrics.Mean(),
            'prediction_latency': tf.keras.metrics.Mean(),
            'optimization_time': tf.keras.metrics.Mean(),
            'fuel_consumption_estimate': tf.keras.metrics.Mean(),
            'trajectory_deviation': tf.keras.metrics.MeanSquaredError(),
            'orbital_stability': tf.keras.metrics.Mean()
        }
        
        # Métricas de Prometheus para exportación
        self.prometheus_metrics = {
            'position_error': Gauge('orbix_position_error', 'Error cuadrático medio en la posición'),
            'velocity_error': Gauge('orbix_velocity_error', 'Error absoluto medio en la velocidad'),
            'collision_probability': Gauge('orbix_collision_probability', 'Probabilidad de colisión'),
            'prediction_latency': Gauge('orbix_prediction_latency_ms', 'Latencia de predicción en ms'),
            'optimization_time': Gauge('orbix_optimization_time_ms', 'Tiempo de optimización en ms'),
            'fuel_consumption_estimate': Gauge('orbix_fuel_consumption', 'Estimación de consumo de combustible'),
            'trajectory_deviation': Gauge('orbix_trajectory_deviation', 'Desviación de la trayectoria óptima'),
            'orbital_stability': Gauge('orbix_orbital_stability', 'Índice de estabilidad orbital')
        }
        
        self.last_update_time = time.time()
    
    def update_metrics(self, predictions, ground_truth):
        """
        Actualiza las métricas con nuevos datos de predicción y valores reales.
        
        Args:
            predictions (tf.Tensor): Tensor con las predicciones del modelo.
            ground_truth (tf.Tensor): Tensor con los valores reales.
        """
        # Registrar el tiempo de inicio para medir la latencia
        start_time = time.time()
        
        # Actualizar métricas de error de posición y velocidad
        self.metrics['position_error'].update_state(predictions, ground_truth)
        self.metrics['velocity_error'].update_state(predictions, ground_truth)
        
        # Calcular y actualizar la latencia
        latency = (time.time() - start_time) * 1000  # Convertir a milisegundos
        self.metrics['prediction_latency'].update_state(latency)
        
        # Actualizar timestamp de última actualización
        self.last_update_time = time.time()
        
        logger.debug("Métricas actualizadas. Latencia: %.2f ms", latency)
    
    def update_collision_probability(self, probability):
        """
        Actualiza la métrica de probabilidad de colisión.
        
        Args:
            probability (float): Valor de probabilidad entre 0 y 1.
        """
        self.metrics['collision_probability'].update_state(probability)
        logger.info("Probabilidad de colisión actualizada: %.4f", probability)
    
    def update_optimization_metrics(self, optimization_time, fuel_consumption=None, 
                                   trajectory_deviation=None, orbital_stability=None):
        """
        Actualiza las métricas relacionadas con la optimización de rutas.
        
        Args:
            optimization_time (float): Tiempo de ejecución del algoritmo de optimización en ms.
            fuel_consumption (float, optional): Estimación del consumo de combustible.
            trajectory_deviation (float, optional): Desviación de la trayectoria óptima.
            orbital_stability (float, optional): Índice de estabilidad orbital entre 0 y 1.
        """
        self.metrics['optimization_time'].update_state(optimization_time)
        
        if fuel_consumption is not None:
            self.metrics['fuel_consumption_estimate'].update_state(fuel_consumption)
        
        if trajectory_deviation is not None:
            self.metrics['trajectory_deviation'].update_state(trajectory_deviation)
        
        if orbital_stability is not None:
            self.metrics['orbital_stability'].update_state(orbital_stability)
        
        logger.debug("Métricas de optimización actualizadas. Tiempo: %.2f ms", optimization_time)
    
    def _update_prometheus_metrics(self):
        """
        Actualiza las métricas de Prometheus con los valores actuales de las métricas de TensorFlow.
        """
        # Transferir valores de métricas internas a Prometheus
        for name, metric in self.metrics.items():
            if name in self.prometheus_metrics:
                self.prometheus_metrics[name].set(metric.result().numpy())
        
        logger.debug("Métricas de Prometheus actualizadas.")
    
    def get_prometheus_metrics(self):
        """
        Obtiene los valores actuales de las métricas para exportación a Prometheus.
        
        Returns:
            dict: Diccionario con los nombres de las métricas y sus valores actuales.
        """
        return {name: metric.result().numpy() for name, metric in self.metrics.items()}
    
    def export_to_prometheus(self, push_gateway_url):
        """
        Exporta las métricas actuales al Push Gateway de Prometheus.
        
        Args:
            push_gateway_url (str): URL del Push Gateway de Prometheus.
        """
        try:
            # Actualizar las métricas de Prometheus
            self._update_prometheus_metrics()
            
            # Exportar a Prometheus
            push_to_gateway(
                push_gateway_url, 
                job='orbix_satellite_metrics', 
                registry=self.prometheus_metrics
            )
            
            logger.info("Métricas exportadas a Prometheus: %s", push_gateway_url)
        except Exception as e:
            logger.error("Error al exportar métricas a Prometheus: %s", str(e))
    
    def reset_metrics(self):
        """
        Reinicia todas las métricas a sus valores iniciales.
        """
        for metric in self.metrics.values():
            metric.reset_states()
        
        logger.info("Métricas reiniciadas.")
    
    def get_metrics_summary(self):
        """
        Genera un resumen de las métricas actuales en formato legible.
        
        Returns:
            str: Resumen de las métricas en formato de texto.
        """
        summary = "Resumen de métricas de telemetría:\n"
        for name, metric in self.metrics.items():
            summary += f"  - {name}: {metric.result().numpy():.6f}\n"
        summary += f"Última actualización: {self.last_update_time}\n"
        
        return summary