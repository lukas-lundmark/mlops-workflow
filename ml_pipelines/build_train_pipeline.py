#!/usr/bin/env python3

import os
from azureml.core.workspace import Workspace
from dotenv import load_dotenv

from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineEndpoint
from azureml.core.runconfig import RunConfiguration
from azureml.core import Dataset
from azureml.data.dataset_consumption_config import DatasetConsumptionConfig

from ml_pipelines.utils import (
    create_or_reuse_environment,
    create_or_reuse_aml_compute,
    create_or_reuse_experiment,
)

load_dotenv()
ws = Workspace.from_config()
instance = create_or_reuse_aml_compute(ws)
target_env = create_or_reuse_environment(ws)
experiment = create_or_reuse_experiment(ws)

# Get preexisting dataset
train_dataset = Dataset.get_by_name(ws, "departure-dataset-train")
test_dataset = Dataset.get_by_name(ws, "departure-dataset-test")

# Create a Dataset Consumption Config for training and test dataset
train_ds_consumption = DatasetConsumptionConfig("train_ds", train_dataset)
test_ds_consumption = DatasetConsumptionConfig("test_ds", test_dataset)
inputs = [train_ds_consumption, test_ds_consumption]

run_config = RunConfiguration()
run_config.environment = target_env

train_step = PythonScriptStep(
    name="test_train_model",
    script_name="train.py",
    source_directory="src",
    compute_target=instance,
    runconfig=run_config,
    inputs=inputs,
    allow_reuse=False,
)

pipeline = Pipeline(
    workspace=ws, steps=[train_step], description="Model Training and Deployment"
)
pipeline.validate()

training_pipeline_name = os.environ.get("TRAIN_PIPELINE_NAME", "training_pipeline")
pipeline_endpoint_name = training_pipeline_name

published_pipeline = pipeline.publish(training_pipeline_name)

try:
    pipeline_endpoint = PipelineEndpoint.get(ws, name=pipeline_endpoint_name)
    pipeline_endpoint.add_default(published_pipeline)
except Exception:
    pipeline_endpoint = PipelineEndpoint.publish(
        workspace=ws,
        name=pipeline_endpoint_name,
        pipeline=published_pipeline,
        description="Pipeline Endpoint for Departure Prediction",
    )
