from __future__ import annotations
from logging import getLogger
from typing import Any

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import complex128, float32, float64, dtype as torch_dtype, device as torch_device
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from qadence.ml_tools.train_utils.strategy import DistributionStrategy
from qadence.ml_tools.data import data_to_device, InfiniteTensorDataset

logger = getLogger("ml_tools")


class Accelerator(DistributionStrategy):
    """
    A class for handling distributed training with PyTorch.

    This class extends `DistributionStrategy` to manage distributed training using PyTorch's
    `torch.distributed` API. It supports spawning multiple processes and wrapping models with
    `DistributedDataParallel` (DDP) when required.

    Attributes:
        spawn (bool): Whether to use multiprocessing spawn mode for process initialization.
        nprocs (int): Number of processes to launch for distributed training.
    """

    def __init__(
        self,
        spawn: bool = False,
        nprocs: int | None = 1,
        compute_setup: str = "auto",
        log_setup: str = "cpu",
        dtype: torch_dtype | None = torch.float32,
        backend: str = "nccl",
    ) -> None:
        """
        Initializes the Accelerator class.

        Args:
            spawn (bool): Whether to use the `spawn` method for multiprocessing. Default is False.
            nprocs (int): Number of processes to launch. Default is 1.
            compute_setup (str): Compute device setup; options are "auto" (default), "gpu", or "cpu".
                - "auto": Uses GPU if available, otherwise CPU.
                - "gpu": Forces GPU usage, raising an error if no CUDA device is available.
                - "cpu": Forces CPU usage.
            log_setup (str): Logging device setup; options are "auto", "cpu" (default).
                - "auto": Uses same device to log as used for computation.
                - "cpu": Forces CPU logging.
            dtype (torch.dtype): Data type for controlling numerical precision. Default is torch.float32.
            backend (str): The backend for distributed communication. Default is "nccl".
        """
        super().__init__(compute_setup, log_setup, dtype, backend)
        self.spawn = spawn
        self.nprocs = nprocs
        self.strategy = self.detect_strategy()
        self._log_warnings()

    def setup(self, process_rank: int | None) -> None:
        """
        Sets up the distributed training environment for a given process.

        Each process sets up a rank, local_rank, and world size. If there are multiple processes
        (based on the world size) a master_add and master port are also assigned.
        Setting up process also sets up the device for the process. These are selected based on 'compute_setup'
        argument in TrainConfig. For compute_setup = "auto" - gpus are selected if available.
        The selected devices could be
            - "cpu": in case of cpu based computation
            - "cuda:n": GPU based on the distributed setup. Note that n is the local_rank of the gpu.
        This also sets up the logging device for each process. In case the log_setup is "auto",
        log_device is the same as device - otherwise its "cpu".

        This method initializes the distributed process group and logs relevant details.

        Args:
            process_rank (int): The rank of the process in the distributed setting.
        """
        self.setup_process(process_rank, self.nprocs)

        logger.info("Initializing Accelerator")
        logger.info("=============================")
        logger.info("  Node             : %s", self.node_name)
        logger.info("  Rank             : %d", self.rank)
        logger.info("  Local Rank       : %d", self.local_rank)
        logger.info("  World Size       : %d", self.world_size)
        logger.info("  Device           : %s", self.device)
        logger.info("  Master Address   : %s", self.master_addr)
        logger.info("  Master Port      : %s", self.master_port)

        self.start_process_group()

    def finalize(self) -> None:
        """Cleans up the distributed process group, ensuring proper shutdown of distributed communication."""
        self.cleanup_process_group()

    def prepare(self, *args: Any) -> tuple[Any, ...]:
        """
        Prepares models, optimizers, and dataloaders for distributed training.

        Moves models to the appropriate device and casts them to the specified precision (`dtype`),
        and wraps them in `DistributedDataParallel` (DDP) if applicable. Adjusts DataLoaders to use
        `DistributedSampler` when running in a distributed setting, and moves data to the correct device
        with the proper precision.
        IMP: We only move the model and optimizer to the device and specified dtypes in the prepare
        For dataloader, we only create a proper sampler, the batch od data should be moved to device
        and dtype seperately.

        Args:
            *args (Any): Objects to be prepared, including models, optimizers, and DataLoaders.

        Returns:
            tuple[Any, ...]: A tuple of prepared objects with necessary modifications for distributed training.
        """
        prepared = []
        for obj in args:
            if isinstance(obj, nn.Module):
                # Move the model to the correct device and cast its parameters to the specified dtype.
                obj = obj.to(device=self.device, dtype=self.dtype)
                if self.world_size and self.world_size > 1:
                    if self.device.startswith("cuda"):  # type: ignore[union-attr]
                        # Wrap the model with DistributedDataParallel for multi-GPU training.
                        obj = DDP(obj, device_ids=[self.local_rank])
                    else:
                        if not self.rank:
                            logger.info("Using CPU for training; skipping DDP wrapping.")
                prepared.append(obj)
            elif isinstance(obj, optim.Optimizer):
                prepared.append(obj)
            elif isinstance(obj, DataLoader):
                if self.world_size and self.world_size > 1:
                    if not isinstance(obj.dataset, InfiniteTensorDataset):
                        sampler = DistributedSampler(
                            obj.dataset, num_replicas=self.world_size, rank=self.local_rank
                        )
                    else:
                        sampler = None
                    obj = DataLoader(
                        obj.dataset,
                        batch_size=obj.batch_size,
                        sampler=sampler,
                        num_workers=getattr(obj, "num_workers", 0),
                        pin_memory=getattr(obj, "pin_memory", False),
                    )
                prepared.append(obj)
            else:
                prepared.append(obj)
        return tuple(prepared)

    def all_reduce_dict(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Performs an all-reduce operation on a dictionary of tensors, averaging values across all processes.

        Args:
            d (dict[str, torch.Tensor]): A dictionary where values are tensors to be reduced across processes.

        Returns:
            dict[str, torch.Tensor]: A dictionary with the reduced tensors, averaged over the world size.
        """
        if dist.is_initialized():
            world_size = dist.get_world_size()
            reduced: dict[str, torch.Tensor] = {}
            for key, tensor in d.items():
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, device=self.device, dtype=self.dtype)
                tensor = tensor.detach().clone()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= world_size
                reduced[key] = tensor
            return reduced
        else:
            return d

    def _log_warnings(self) -> None:
        if self.spawn:
            if self.strategy == "torchrun":
                logger.warning(
                    f"Spawn mode is enabled (spawn={self.spawn}), but the process was launched using `torchrun`, "
                    "which is incompatible with spawning new processes."
                )
                logger.warning("Setting spawn=False")
                self.spawn = False
