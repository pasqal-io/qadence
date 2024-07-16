from __future__ import annotations

import os
import shutil
from itertools import count
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import mlflow
import pytest
import torch
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mlflow import MlflowClient
from mlflow.entities import Run
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from qadence.ml_tools import TrainConfig, train_with_grad
from qadence.ml_tools.data import to_dataloader
from qadence.ml_tools.utils import rand_featureparameters
from qadence.models import QNN, QuantumModel
from qadence.types import ExperimentTrackingTool


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


def setup(model: Module) -> tuple[Callable, Optimizer]:
    cnt = count()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    inputs = rand_featureparameters(model, 1)

    def loss_fn(model: QuantumModel, data: torch.Tensor) -> tuple[torch.Tensor, dict]:
        next(cnt)
        out = model.expectation(inputs)
        loss = criterion(out, torch.rand(1))
        return loss, {}

    return loss_fn, optimizer


def load_mlflow_model(train_config: TrainConfig) -> None:
    run_id = train_config.mlflow_config.run.info.run_id

    mlflow.pytorch.load_model(model_uri=f"runs:/{run_id}/model")


def clean_mlflow_experiment(train_config: TrainConfig) -> None:
    experiment_id = train_config.mlflow_config.run.info.experiment_id
    client = MlflowClient()

    runs = client.search_runs(experiment_id)

    def clean_artifacts(run: Run) -> None:
        artifact_uri = run.info.artifact_uri
        parsed_uri = urlparse(artifact_uri)
        local_path = os.path.abspath(os.path.join(parsed_uri.netloc, parsed_uri.path))
        shutil.rmtree(local_path)

    for run in runs:
        clean_artifacts(run)

        run_id = run.info.run_id
        client.delete_run(run_id)

        mlruns_base_dir = "./mlruns"
        if os.path.isdir(mlruns_base_dir):
            shutil.rmtree(os.path.join(mlruns_base_dir, experiment_id))


def test_hyperparams_logging_mlflow(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    model = BasicQuantumModel

    loss_fn, optimizer = setup(model)

    hyperparams = {"max_iter": int(10), "lr": 0.1}

    config = TrainConfig(
        folder=tmp_path,
        max_iter=hyperparams["max_iter"],  # type: ignore
        checkpoint_every=1,
        write_every=1,
        hyperparams=hyperparams,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    train_with_grad(model, None, optimizer, config, loss_fn=loss_fn)

    mlflow_config = config.mlflow_config
    experiment_id = mlflow_config.run.info.experiment_id
    run_id = mlflow_config.run.info.run_id

    experiment_dir = Path(f"mlruns/{experiment_id}")
    hyperparams_files = [experiment_dir / run_id / "params" / key for key in hyperparams.keys()]

    assert all([os.path.isfile(hf) for hf in hyperparams_files])

    clean_mlflow_experiment(config)


def test_hyperparams_logging_tensorboard(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    model = BasicQuantumModel

    loss_fn, optimizer = setup(model)

    hyperparams = {"max_iter": int(10), "lr": 0.1}

    config = TrainConfig(
        folder=tmp_path,
        max_iter=hyperparams["max_iter"],  # type: ignore
        checkpoint_every=1,
        write_every=1,
        hyperparams=hyperparams,
        tracking_tool=ExperimentTrackingTool.TENSORBOARD,
    )

    train_with_grad(model, None, optimizer, config, loss_fn=loss_fn)


def test_model_logging_mlflow_basicQM(BasicQuantumModel: QuantumModel, tmp_path: Path) -> None:
    model = BasicQuantumModel

    loss_fn, optimizer = setup(model)

    config = TrainConfig(
        folder=tmp_path,
        max_iter=10,  # type: ignore
        checkpoint_every=1,
        write_every=1,
        log_model=True,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    train_with_grad(model, None, optimizer, config, loss_fn=loss_fn)

    load_mlflow_model(config)

    clean_mlflow_experiment(config)


def test_model_logging_mlflow_basicQNN(BasicQNN: QNN, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQNN

    loss_fn, optimizer = setup(model)

    config = TrainConfig(
        folder=tmp_path,
        max_iter=10,  # type: ignore
        checkpoint_every=1,
        write_every=1,
        log_model=True,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)

    load_mlflow_model(config)

    clean_mlflow_experiment(config)


def test_model_logging_mlflow_basicAdjQNN(BasicAdjointQNN: QNN, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicAdjointQNN

    loss_fn, optimizer = setup(model)

    config = TrainConfig(
        folder=tmp_path,
        max_iter=10,  # type: ignore
        checkpoint_every=1,
        write_every=1,
        log_model=True,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
    )

    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)

    load_mlflow_model(config)

    clean_mlflow_experiment(config)


def test_model_logging_tensorboard(
    BasicQuantumModel: QuantumModel, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    model = BasicQuantumModel

    loss_fn, optimizer = setup(model)

    config = TrainConfig(
        folder=tmp_path,
        max_iter=10,  # type: ignore
        checkpoint_every=1,
        write_every=1,
        log_model=True,
        tracking_tool=ExperimentTrackingTool.TENSORBOARD,
    )

    train_with_grad(model, None, optimizer, config, loss_fn=loss_fn)

    assert "Model logging is not supported by tensorboard. No model will be logged." in caplog.text


def test_plotting_mlflow(BasicQNN: QNN, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQNN

    loss_fn, optimizer = setup(model)

    def plot_model(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
        descr = f"model_prediction_epoch_{iteration}.png"
        fig, ax = plt.subplots()
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        out = model.expectation(x)
        ax.plot(x.detach().numpy(), out.detach().numpy())
        return descr, fig

    def plot_error(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
        descr = f"error_epoch_{iteration}.png"
        fig, ax = plt.subplots()
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        out = model.expectation(x)
        ground_truth = torch.rand_like(out)
        error = ground_truth - out
        ax.plot(x.detach().numpy(), error.detach().numpy())
        return descr, fig

    config = TrainConfig(
        folder=tmp_path,
        max_iter=10,
        checkpoint_every=1,
        write_every=1,
        tracking_tool=ExperimentTrackingTool.MLFLOW,
        plotting_functions=(plot_model, plot_error),
    )

    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)

    clean_mlflow_experiment(config)


def test_plotting_tensorboard(BasicQNN: QNN, tmp_path: Path) -> None:
    data = dataloader()
    model = BasicQNN

    loss_fn, optimizer = setup(model)

    def plot_model(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
        descr = f"model_prediction_epoch_{iteration}.png"
        fig, ax = plt.subplots()
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        out = model.expectation(x)
        ax.plot(x.detach().numpy(), out.detach().numpy())
        return descr, fig

    def plot_error(model: QuantumModel, iteration: int) -> tuple[str, Figure]:
        descr = f"error_epoch_{iteration}.png"
        fig, ax = plt.subplots()
        x = torch.linspace(0, 1, 100).reshape(-1, 1)
        out = model.expectation(x)
        ground_truth = torch.rand_like(out)
        error = ground_truth - out
        ax.plot(x.detach().numpy(), error.detach().numpy())
        return descr, fig

    config = TrainConfig(
        folder=tmp_path,
        max_iter=10,
        checkpoint_every=1,
        write_every=1,
        tracking_tool=ExperimentTrackingTool.TENSORBOARD,
        plotting_functions=(plot_model, plot_error),
    )

    train_with_grad(model, data, optimizer, config, loss_fn=loss_fn)
