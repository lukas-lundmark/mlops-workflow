#!/usr/bin/env python3

import os
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.experiment import Experiment


def create_or_reuse_aml_compute(ws) -> ComputeTarget:
    """Create a CPU or reuse AML Compute Cluster

    You can set the name of cpu cluster with CPUCLUSTER Environment variable

    Args:
      ws: AML Workspace Instance
    """
    cpu_cluster_name = os.environ.get("CPUCLUSTER", "cpucluster")

    # Verify that cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
        print("Found existing cluster, use it.")
    except ComputeTargetException:
        # To use a different region for the compute, add a location='<region>' parameter
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_D2_V2",
            max_nodes=4,
            idle_seconds_before_scaledown=300,
        )
        cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)
    return cpu_cluster


def create_or_reuse_environment(ws) -> Environment:
    """Create or reuse an AML environment

    You can set the environment name via the ENVIRONMENT_NAME env variable

    Args:
      ws: AML Workspace Instance
    """
    environment_name = os.environ.get("ENVIRONMENT_NAME", "my-conda")
    try:
        env = Environment.get(ws, name=environment_name)
    except Exception:
        env = Environment.from_conda_specification(
            name=environment_name, file_path="environment_setup/ci_dependencies.yml"
        )

    return env


def create_or_reuse_experiment(ws) -> Experiment:
    experiment_name = os.environ.get("EXPERIMENT_NAME", "train_pipeline")
    return Experiment(ws, name=experiment_name)
