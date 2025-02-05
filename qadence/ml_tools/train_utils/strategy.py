from __future__ import annotations
from logging import getLogger
from typing import Tuple

import os
import subprocess
import torch
import torch.distributed as dist

logger = getLogger("ml_tools")


class DistributionStrategy:
    """
    A class to handle the configuration and initialization of the PyTorch distributed.

    process group based on the launch environment (e.g., torchrun, SLURM, or none).

    This class auto-detects the launch strategy by examining environment variables.
    It ensures that the MASTER_ADDR and MASTER_PORT environment variables are set,
    and it stores global distributed attributes as instance attributes.

    Attributes:
        backend (str): The backend to use for distributed training ("nccl", "gloo", etc.).
        strategy (str): The detected launch strategy ("torchrun", "slurm", or "none").
        rank (int): The global rank of the current process.
        world_size (int): The total number of processes.
        local_rank (int): The rank of the process on its local node.
        master_addr (str | None): The master node address used for distributed training.
        master_port (str | None): The master node port used for distributed training.
    """

    def __init__(self, backend: str = "nccl") -> None:
        """
        Initialize the DistributionStrategy instance.

        Args:
            backend (str): The backend to use for distributed training.
                           Common options include "nccl" (recommended for GPUs)
                           or "gloo" (can be used for CPUs).
        """
        self.backend: str = backend
        self.strategy: str = self.detect_strategy()
        self.rank: int | None = None
        self.world_size: int | None = None
        self.local_rank: int | None = None
        self.master_addr: str | None = None
        self.master_port: str | None = None

    def detect_strategy(self) -> str:
        """
        Detect the current launch strategy based on environment variables.

        Returns:
            str: The detected strategy:
                 - "torchrun" if LOCAL_RANK is present,
                 - "slurm" if SLURM_PROCID is present,
                 - "none" otherwise.
        """
        if "LOCAL_RANK" in os.environ:
            return "torchrun"
        elif "SLURM_PROCID" in os.environ:
            return "slurm"
        else:
            return "none"

    @staticmethod
    def get_master_addr_port(strategy: str) -> Tuple[str, str]:
        """
        Determine and set the MASTER_ADDR and MASTER_PORT environment variables needed for distributed training.

        For a SLURM-based launch, uses SLURM_NODELIST to determine the master node.
        Otherwise, defaults to "localhost" and port "12355".

        Args:
            strategy (str): The launch strategy ("slurm", "torchrun", or "none").

        Returns:
            Tuple[str, str]: The master node address and port.
        """
        if "MASTER_ADDR" in os.environ:
            master_addr: str = os.environ["MASTER_ADDR"]
        else:
            if strategy == "slurm":
                nodelist: str | None = os.environ.get("SLURM_NODELIST")
                if nodelist:
                    try:
                        master_addr = (
                            subprocess.check_output(["scontrol", "show", "hostnames", nodelist])
                            .splitlines()[0]
                            .decode("utf-8")
                            .strip()
                        )
                    except Exception:
                        master_addr = "localhost"
                else:
                    master_addr = "localhost"
            else:
                master_addr = "localhost"
            os.environ["MASTER_ADDR"] = master_addr

        if "MASTER_PORT" in os.environ:
            master_port: str = os.environ["MASTER_PORT"]
        else:
            master_port = "12355"
            os.environ["MASTER_PORT"] = master_port

        return master_addr, master_port

    def set_attributes(self) -> Tuple[int, int, int]:
        """
        Set the distributed attributes (rank, world_size, local_rank) from environment variables.

        This method does not initialize the process group.

        Returns:
            Tuple[int, int, int]: A tuple (rank, world_size, local_rank).
        """
        if self.strategy == "slurm":
            rank: int = int(os.environ["SLURM_PROCID"])
            world_size: int = int(os.environ["SLURM_NTASKS"])
            local_rank: int = int(os.environ["SLURM_LOCALID"])
        elif self.strategy == "torchrun":
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", rank))
        else:
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            local_rank = 0

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank

        # If needed, obtain master_addr and master_port.
        if world_size > 1:
            master_addr, master_port = DistributionStrategy.get_master_addr_port(self.strategy)
            self.master_addr = master_addr
            self.master_port = master_port
            logger.info("MASTER_ADDR: %s, MASTER_PORT: %s", master_addr, master_port)

        logger.info("Strategy: %s", self.strategy)
        logger.info("Rank: %d, World Size: %d, Local Rank: %d", rank, world_size, local_rank)
        return rank, world_size, local_rank

    def start(self) -> None:
        """
        Initializes the PyTorch distributed process group using the stored attributes.

        This function should be called after set_attributes().
        """
        if self.world_size is not None and self.world_size > 1:
            # Initialize the process group with the backend, rank, and world size.
            dist.init_process_group(
                backend=self.backend, rank=self.rank, world_size=self.world_size
            )
            logger.info("Initialized process group with backend '%s'", self.backend)
        else:
            logger.info("Process group initialization skipped (world_size <= 1)")

    def cleanup(self) -> None:
        """Cleans up the distributed process group by destroying it if it is initialized."""
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed distributed process group.")
