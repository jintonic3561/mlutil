# -*- coding: utf-8 -*-
"""
Created on Sun May 22 12:38:32 2022

@author: jintonic
"""

import os

try:
    import mlflow
except:
    print("Cannot import mlflow.")


def run(
    experiment_name: str,
    run_name: str,
    params: dict,
    metrics: dict,
    artifact_paths: list,
    uri: str = None,
):
    if not uri:
        uri = "mlflow/mlruns"
    
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for i in artifact_paths:
            mlflow.log_artifact(i)
