from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader

from qadence.ml_tools import TrainConfig, Trainer
from qadence.ml_tools.callbacks import (
    EarlyStopping,
    GradientMonitoring,
    LoadCheckpoint,
    LogHyperparameters,
    LogModelTracker,
    LRSchedulerCosineAnnealing,
    LRSchedulerCyclic,
    LRSchedulerStepDecay,
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
    writer.write.assert_called_once_with(trainer.opt_result.iteration, trainer.opt_result.metrics)


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


def test_lr_scheduler_step_decay(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    initial_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    decay_factor = 0.5
    callback = LRSchedulerStepDecay(on=stage, called_every=1, gamma=decay_factor)
    callback(stage, trainer, trainer.config, writer)

    new_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    assert new_lr == initial_lr * decay_factor


def test_lr_scheduler_cyclic(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    base_lr = 0.001
    max_lr = 0.01
    step_size = 2000
    callback = LRSchedulerCyclic(
        on=stage, called_every=1, base_lr=base_lr, max_lr=max_lr, step_size=step_size
    )

    # Set trainer's iteration to simulate training progress
    trainer.opt_result.iteration = step_size // 2  # Middle of the cycle
    callback(stage, trainer, trainer.config, writer)
    expected_lr = base_lr + (max_lr - base_lr) * 0.5
    new_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    assert math.isclose(new_lr, expected_lr, rel_tol=1e-6)


def test_lr_scheduler_cosine_annealing(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    min_lr = 1e-6
    t_max = 5000
    initial_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    callback = LRSchedulerCosineAnnealing(on=stage, called_every=1, t_max=t_max, min_lr=min_lr)

    trainer.opt_result.iteration = t_max // 2  # Halfway through the cycle
    callback(stage, trainer, trainer.config, writer)

    expected_lr = min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * 0.5)) / 2
    new_lr = trainer.optimizer.param_groups[0]["lr"]  # type: ignore
    assert math.isclose(new_lr, expected_lr, rel_tol=1e-6)


def test_early_stopping(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    patience = 2
    monitor_metric = "val_loss"
    mode = "min"
    callback = EarlyStopping(
        on=stage, called_every=1, monitor=monitor_metric, patience=patience, mode=mode
    )

    # Simulate metric values
    trainer.opt_result.metrics = {monitor_metric: 0.5}
    callback(stage, trainer, trainer.config, writer)
    assert trainer._stop_training.detach().item() == 0

    trainer.opt_result.metrics[monitor_metric] = 0.6
    callback(stage, trainer, trainer.config, writer)
    assert trainer._stop_training.detach().item() == 0

    trainer.opt_result.metrics[monitor_metric] = 0.7
    callback(stage, trainer, trainer.config, writer)
    assert (
        trainer._stop_training.detach().item() == 1
    )  # Should stop training after patience exceeded


def test_gradient_monitoring(trainer: Trainer) -> None:
    writer = trainer.callback_manager.writer = Mock()
    stage = trainer.training_stage
    callback = GradientMonitoring(on=stage, called_every=1)

    for param in trainer.model.parameters():
        param.grad = torch.ones_like(param) * 0.1

    callback(stage, trainer, trainer.config, writer)
    expected_keys = {
        f"{name}_{stat}"
        for name, param in trainer.model.named_parameters()
        for stat in ["mean", "std", "max", "min"]
    }

    written_keys = writer.write.call_args[0][1].keys()
    assert set(written_keys) == expected_keys
