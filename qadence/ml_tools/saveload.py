from __future__ import annotations

import os
import re
from logging import getLogger
from pathlib import Path
from typing import Any

import torch
from nevergrad.optimization.base import Optimizer as NGOptimizer
from torch.nn import Module
from torch.optim import Optimizer

logger = getLogger(__name__)


def get_latest_checkpoint_name(folder: Path, type: str, device: str | torch.device = "cpu") -> Path:
    file = Path("")
    files = [f for f in os.listdir(folder) if f.endswith(".pt") and type in f]
    if len(files) == 0:
        logger.error(f"Directory {folder} does not contain any {type} checkpoints.")
    if len(files) == 1:
        file = Path(files[0])
    else:
        device = str(device).split(":")[0]
        pattern = re.compile(f".*_(\d+)_device_{device}.pt$")
        legacy_pattern = re.compile(".*_(\d+).pt$")
        max_index = -1
        for f in files:
            legacy_match = legacy_pattern.search(f)
            match = pattern.search(f)
            if match or legacy_match:
                if legacy_match:
                    logger.warn(f"Found checkpoint(s) in legacy format: {f}.")
                    match = legacy_match
                index_str = match.group(1).replace("_", "")  # type: ignore [union-attr]
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    file = Path(f)
    return Path(file)


def load_checkpoint(
    folder: Path,
    model: Module,
    optimizer: Optimizer | NGOptimizer,
    model_ckpt_name: str | Path = "",
    opt_ckpt_name: str | Path = "",
    device: str | torch.device = "cpu",
) -> tuple[Module, Optimizer | NGOptimizer, int]:
    if isinstance(folder, str):
        folder = Path(folder)
    if not folder.exists():
        folder.mkdir(parents=True)
        return model, optimizer, 0
    model, iter = load_model(folder, model, model_ckpt_name, device)
    optimizer = load_optimizer(folder, optimizer, opt_ckpt_name, device)
    return model, optimizer, iter


def write_checkpoint(
    folder: Path,
    model: Module,
    optimizer: Optimizer | NGOptimizer,
    iteration: int | str,
) -> None:
    from qadence import QuantumModel

    from .models import QNN

    device = None
    try:
        # We extract the device from the pyqtorch native circuit
        device = (
            model.device
            if isinstance(model, (QNN, QuantumModel))
            else next(model.parameters()).device
        )
        device = str(device).split(":")[0]  # in case of using several CUDA devices
    except Exception as e:
        msg = (
            f"Unable to identify in which device the QuantumModel is stored due to {e}."
            "Setting device to None"
        )
        logger.warning(msg)

    iteration_substring = f"{iteration:03n}" if isinstance(iteration, int) else iteration
    model_checkpoint_name: str = (
        f"model_{type(model).__name__}_ckpt_"
        + f"{iteration_substring}"
        + f"_device_{device}"
        + ".pt"
    )
    opt_checkpoint_name: str = (
        f"opt_{type(optimizer).__name__}_ckpt_"
        + f"{iteration_substring}"
        + f"_device_{device}"
        + ".pt"
    )
    try:
        d = (
            model._to_dict(save_params=True)
            if isinstance(model, (QNN, QuantumModel))
            else model.state_dict()
        )
        torch.save((iteration, d), folder / model_checkpoint_name)
        logger.info(f"Writing {type(model).__name__} checkpoint {model_checkpoint_name}")
    except Exception as e:
        logger.exception(e)
    try:
        if isinstance(optimizer, Optimizer):
            torch.save(
                (iteration, type(optimizer), optimizer.state_dict()), folder / opt_checkpoint_name
            )
        elif isinstance(optimizer, NGOptimizer):
            optimizer.dump(folder / opt_checkpoint_name)
        logger.info(f"Writing {type(optimizer).__name__} to checkpoint {opt_checkpoint_name}")
    except Exception as e:
        logger.exception(e)


def load_model(
    folder: Path,
    model: Module,
    model_ckpt_name: str | Path = "",
    device: str | torch.device = "cpu",
    *args: Any,
    **kwargs: Any,
) -> tuple[Module, int]:
    from qadence import QNN, QuantumModel

    iteration = 0
    if model_ckpt_name == "":
        model_ckpt_name = get_latest_checkpoint_name(folder, "model", device)

    try:
        iteration, model_dict = torch.load(folder / model_ckpt_name, *args, **kwargs)
        if isinstance(model, (QuantumModel, QNN)):
            model.load_params_from_dict(model_dict)
        elif isinstance(model, Module):
            model.load_state_dict(model_dict, strict=True)
        # Load model to a specific gpu device if specified
        pattern = re.compile("cuda:\d+$")
        if pattern.search(str(device)):
            model.to(device)

    except Exception as e:
        msg = f"Unable to load state dict due to {e}.\
               No corresponding pre-trained model found. Returning the un-trained model."
        logger.warning(msg)
    return model, iteration


def load_optimizer(
    folder: Path,
    optimizer: Optimizer | NGOptimizer,
    opt_ckpt_name: str | Path = "",
    device: str | torch.device = "cpu",
) -> Optimizer | NGOptimizer:
    if opt_ckpt_name == "":
        opt_ckpt_name = get_latest_checkpoint_name(folder, "opt", device)
    if os.path.isfile(folder / opt_ckpt_name):
        if isinstance(optimizer, Optimizer):
            (_, OptType, optimizer_state) = torch.load(folder / opt_ckpt_name)
            if isinstance(optimizer, OptType):
                optimizer.load_state_dict(optimizer_state)

        elif isinstance(optimizer, NGOptimizer):
            loaded_optimizer = NGOptimizer.load(folder / opt_ckpt_name)
            if loaded_optimizer.name == optimizer.name:
                optimizer = loaded_optimizer
        else:
            raise NotImplementedError
    return optimizer
