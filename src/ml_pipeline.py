import logging
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, Transform, Trainer, Pusher
from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from tfx.proto import trainer_pb2, pusher_pb2

logger = logging.getLogger("Orbix")

def run_tfx_pipeline():
    """
    Ejecuta un pipeline de ML utilizando TFX.
    """
    logger.info("Iniciando pipeline TFX...")
    context = InteractiveContext()
    
    example_gen = CsvExampleGen(input_base='data/')
    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'], infer_feature_shape=True)
    transform = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file='preprocessing.py'
    )
    trainer = Trainer(
        module_file='trainer.py',
        examples=transform.outputs['transformed_examples'],
        transform_graph=transform.outputs['transform_graph'],
        schema=schema_gen.outputs['schema'],
        train_args=trainer_pb2.TrainArgs(num_steps=10000),
        eval_args=trainer_pb2.EvalArgs(num_steps=5000)
    )
    pusher = Pusher(
        model=trainer.outputs['model'],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory='serving_model')
        )
    )
    context.run()
