from __future__ import annotations
from logging import getLogger
from typing import Any, Tuple

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from qadence.ml_tools.train_utils.strategy import DistributionStrategy

logger = getLogger("ml_tools")


class Accelerator:
    """
    A Accelerator class to prepare objects for distributed training and to run the training loop.

    This class leverages a DistributionStrategy instance (accessed as `self.dist`)
    to configure the distributed environment. It moves models to the appropriate device
    and wraps them with DistributedDataParallel (if needed), and optionally spawns multiple
    processes using torch.multiprocessing.

    The global distributed attributes (rank, local_rank, world_size, master_addr, master_port)
    are maintained solely by the DistributionStrategy instance.

    Attributes:
        backend (str): The distributed backend used (e.g., "nccl" or "gloo").
        world_size (int): The total number of processes intended to run (provided by the user).
        spawn (bool): Whether to spawn processes using torch.multiprocessing.
        dist (DistributionStrategy): The distribution strategy instance containing global attributes.
        device (str): The device string (e.g., "cuda:0") for the current process.
    """

    def __init__(
        self, backend: str = "nccl", world_size: int | None = 1, spawn: bool = False
    ) -> None:
        """
        Initialize the Accelerator.

        Args:
            backend (str): The distributed backend to use ("nccl" or "gloo").
            world_size (int): The total number of processes for distributed training.
                              This value will override the world size detected from the environment.
            spawn (bool): Whether to spawn processes using torch.multiprocessing.
        """
        self.backend: str = backend
        self.spawn: bool = spawn
        self.world_size: int | None
        # Create a DistributionStrategy instance to set up the distributed environment.
        self.dist = DistributionStrategy(backend)
        rank, env_world_size, local_rank = self.dist.set_attributes()
        if world_size and str(world_size) != str(env_world_size):
            logger.warning(
                "Provided world size (%d) does not match environment world size (%d). Using provided world size.",
                world_size,
                env_world_size,
            )
            self.dist.world_size = self.world_size = world_size
        else:
            self.dist.world_size = self.world_size = env_world_size
        self.dist.start()

        self.device: str = f"cuda:{self.dist.local_rank}" if torch.cuda.is_available() else "cpu"
        logger.info(
            "Accelerator initialized: Rank %d, World Size: %d, Local Rank: %d, Device %s",
            self.dist.rank,
            self.dist.world_size,
            self.dist.local_rank,
            self.device,
        )

    def prepare(self, *args: Any) -> Tuple[Any, ...]:
        """
        Prepares and returns a tuple of objects (e.g., model, optimizer, dataloaders) for distributed training.

        For nn.Module objects, moves them to the appropriate device and wraps them with DistributedDataParallel.
        For DataLoader objects, if distributed training is active, wraps the dataset with a DistributedSampler.

        Args:
            *args (Any): A variable number of objects to be prepared.

        Returns:
            Tuple[Any, ...]: A tuple containing the prepared objects.
        """
        prepared = []
        for obj in args:
            if isinstance(obj, nn.Module):
                obj = obj.to(self.device)
                if self.dist.world_size and self.dist.world_size > 1:
                    gpu_id = int(self.device.split(":")[-1])
                    obj = DDP(obj, device_ids=[gpu_id] if torch.cuda.is_available() else None)
                prepared.append(obj)
            elif isinstance(obj, optim.Optimizer):
                prepared.append(obj)
            elif isinstance(obj, DataLoader):
                if self.dist.world_size and self.dist.world_size > 1:
                    sampler = DistributedSampler(
                        obj.dataset, num_replicas=self.dist.world_size, rank=self.dist.rank
                    )
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
        logger.info("Prepared %d objects for distributed training", len(prepared))
        return tuple(prepared)

    def finalize(self) -> None:
        """Finalizes the distributed training by cleaning up the process group."""
        self.dist.cleanup()
        logger.info("Finalized distributed training and cleaned up process group.")

    def run(self, trainer_instance: Any) -> None:
        """
        Runs the training process using the provided trainer instance.

        If self.spawn is True, multiple processes are spawned using torch.multiprocessing.
        Otherwise, the trainer_instance's _prepare() and _fit() methods are called directly.

        Args:
            trainer_instance (Any): An instance that implements _prepare() and _fit() methods.
        """
        if self.spawn:

            def _worker(rank: int) -> None:
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(self.world_size)
                os.environ["LOCAL_RANK"] = str(rank)
                logger.info("Worker process %d starting", rank)
                trainer_instance._prepare()
                trainer_instance._fit()

            mp.spawn(_worker, nprocs=self.world_size)
        else:
            logger.info(
                "Running training in the current process without spawning additional processes."
            )
            trainer_instance._prepare()
            trainer_instance._fit()

    def all_reduce_dict(self, d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Applies an all-reduce operation to a dictionary of tensors, averaging the values across all processes.

        Args:
            d (dict[str, torch.Tensor]): A dictionary where each value is a tensor.

        Returns:
            dict[str, torch.Tensor]: A dictionary with the same keys where each tensor is averaged over all processes.
        """
        if dist.is_initialized():
            world_size = dist.get_world_size()
            reduced = {}
            for key, tensor in d.items():
                # Convert to tensor if not already
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, device=self.device, dtype=torch.float32)
                tensor = tensor.detach().clone()
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= world_size
                reduced[key] = tensor
            return reduced
        else:
            return d
