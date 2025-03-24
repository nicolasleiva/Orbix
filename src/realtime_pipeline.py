import json
import logging
from confluent_kafka import Producer, Consumer
from .config import KAFKA_CONFIG

logger = logging.getLogger("Orbix")

class RealTimeDataPipeline:
    """
    Pipeline para ingestar datos TLE vía Kafka.
    """
    def __init__(self):
        self.producer = Producer(KAFKA_CONFIG)
        self.consumer = Consumer(KAFKA_CONFIG)

    def stream_tle_data(self):
        """
        Generador que emite datos TLE del tópico 'tle-updates'.
        """
        self.consumer.subscribe(['tle-updates'])
        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error("Error en Kafka: %s", msg.error())
                continue
            yield json.loads(msg.value())

    def publish_prediction(self, prediction: dict):
        """
        Publica la predicción en el tópico 'trajectory-predictions'.
        """
        self.producer.produce(
            'trajectory-predictions',
            key=str(prediction.get('satellite_id', 'unknown')),
            value=json.dumps(prediction)
        )
        self.producer.flush()
