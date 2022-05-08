import os
from azureml.core import Experiment, Workspace
from azureml.pipeline.core import PipelineEndpoint

from dotenv import load_dotenv

load_dotenv()
ws = Workspace.from_config()
pipeline_name = os.environ.get("TRAIN_PIPELINE_NAME", "train-pipeline")
experiment_name = os.environ.get("EXPERIMENT_NAME", "demo-train-pipeline-experiment")
pipeline_endpoint = PipelineEndpoint.get(ws, name=pipeline_name)
experiment = Experiment(ws, experiment_name)
run = experiment.submit(
    pipeline_endpoint, tags={"endpoint_version": pipeline_endpoint.default_version}
)
run.wait_for_completion(show_output=True)
