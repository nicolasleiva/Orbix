import tensorflow as tf

class TelemetryMonitor:
    """
    Monitorea métricas en tiempo real de la predicción.
    """
    def __init__(self):
        self.metrics = {
            'position_error': tf.keras.metrics.MeanSquaredError(),
            'velocity_error': tf.keras.metrics.MeanAbsoluteError()
        }

    def update_metrics(self, predictions, ground_truth):
        for metric in self.metrics.values():
            metric.update_state(predictions, ground_truth)

    def get_prometheus_metrics(self):
        return {name: metric.result().numpy() for name, metric in self.metrics.items()}
