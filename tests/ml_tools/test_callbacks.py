from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

from qadence.ml_tools import TrainConfig, Trainer
from qadence.ml_tools.callbacks import (
    LoadCheckpoint,
    LogHyperparameters,
    LogModelTracker,
    PlotMetrics,
    PrintMetrics,
    SaveBestCheckpoint,
    SaveCheckpoint,
    WriteMetrics,
)
from qadence.ml_tools.callbacks.saveload import write_checkpoint
from qadence.ml_tools.data import OptimizeResult, to_dataloader
from qadence.ml_tools.stages import TrainingStage


def dataloader(batch_size: int = 25) -> DataLoader:
    x = torch.linspace(0, 1, batch_size).reshape(-1, 1)
    y = torch.cos(x)
    return to_dataloader(x, y, batch_size=batch_size, infinite=True)


@pytest.fixture
def trainer(Basic: torch.nn.Module, tmp_path: Path) -> Trainer:
    """Set up a real Trainer with a Basic and optimizer."""
    data = dataloader()
    model = Basic
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    config = TrainConfig(
        log_folder=tmp_path,
        max_iter=1,
        checkpoint_best_only=True,
        validation_criterion=lambda loss, best, ep: loss < (best - ep),
        val_epsilon=1e-5,
    )
    trainer = Trainer(
        model=model, optimizer=optimizer, config=config, loss_fn="mse", train_dataloader=data
    )
    trainer.opt_result = OptimizeResult(
        iteration=1,
        model=model,
        optimizer=optimizer,
        loss=torch.tensor(0.5),
        metrics={"accuracy": torch.tensor(0.8)},
    )
    trainer.training_stage = TrainingStage("train_start")
    return trainer


def test_save_checkpoint(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = SaveCheckpoint(stage, called_every=1)
    callback(stage, trainer, trainer.config, writer)

    checkpoint_file = (
        trainer.config.log_folder / f"model_{type(trainer.model).__name__}_ckpt_001_device_cpu.pt"
    )
    assert checkpoint_file.exists()


def test_save_best_checkpoint(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = SaveBestCheckpoint(on=stage, called_every=1)
    callback(stage, trainer, trainer.config, writer)

    best_checkpoint_file = (
        trainer.config.log_folder / f"model_{type(trainer.model).__name__}_ckpt_best_device_cpu.pt"
    )
    assert best_checkpoint_file.exists()
    assert callback.best_loss == trainer.opt_result.loss


def test_print_metrics(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = PrintMetrics(on=stage, called_every=1)
    callback(stage, trainer, trainer.config, writer)
    writer.print_metrics.assert_called_once_with(trainer.opt_result)


def test_write_metrics(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = WriteMetrics(on=stage, called_every=1)
    callback(stage, trainer, trainer.config, writer)
    writer.write.assert_called_once_with(trainer.opt_result)


def test_plot_metrics(trainer: Trainer) -> None:
    trainer.config.plotting_functions = (lambda model, iteration: ("plot_name", None),)
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = PlotMetrics(stage, called_every=1)
    callback(stage, trainer, trainer.config, writer)

    writer.plot.assert_called_once_with(
        trainer.model,
        trainer.opt_result.iteration,
        trainer.config.plotting_functions,
    )


def test_log_hyperparameters(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    trainer.config.hyperparams = {"learning_rate": 0.01, "epochs": 10}
    callback = LogHyperparameters(stage, called_every=1)
    callback(stage, trainer, trainer.config, writer)
    writer.log_hyperparams.assert_called_once_with(trainer.config.hyperparams)


def test_load_checkpoint(trainer: Trainer) -> None:
    # Prepare a checkpoint
    write_checkpoint(trainer.config.log_folder, trainer.model, trainer.optimizer, iteration=1)
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = LoadCheckpoint(stage, called_every=1)
    model, optimizer, iteration = callback(stage, trainer, trainer.config, writer)

    assert model is not None
    assert optimizer is not None
    assert iteration == 1


def test_log_model_tracker(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    callback = LogModelTracker(on=trainer.training_stage, called_every=1)
    callback(trainer.training_stage, trainer, trainer.config, writer)
    writer.log_model.assert_called_once_with(
        trainer.model,
        trainer.train_dataloader,
        trainer.val_dataloader,
        trainer.test_dataloader,
    )
