from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from logging import getLogger

import torch

from qadence.types import ExecutionType

logger = getLogger("ml_tools")


class BaseExecution(ABC):
    """
    Attributes:

       backend (str): The backend used for distributed communication (e.g., "nccl", "gloo").
           It should be one of the backends supported by torch.distributed
       compute_setup (str): Desired computation device setup.
       log_setup (str): Desired logging device setup.
       device (str | None): Computation device, e.g., "cpu" or "cuda:<local_rank>".
       log_device (str | None): Logging device, e.g., "cpu" or "cuda:<local_rank>".
       dtype (torch.dtype): Data type for controlling numerical precision (e.g., torch.float32).
       data_dtype (torch.dtype): Data type for controlling datasets precision (e.g., torch.float16).
       node_rank (int): Rank of the node on the cluster setup.
    """

    def __init__(
        self,
        compute_setup: str = "auto",
        log_setup: str = "cpu",
        backend: str = "nccl",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.compute_setup = compute_setup
        self.log_setup = log_setup
        self.backend = backend
        self.compute: str
        self.device: str
        self.log_device: str

        self.dtype: torch.dtype | None = dtype
        self.data_dtype: torch.dtype | None = None
        # TODO: This is legacy behavior of data_dtype. Modify it appropriately.
        if self.dtype:
            self.data_dtype = torch.float64 if (self.dtype == torch.complex128) else self.dtype

        self._set_cluster_variables()
        self._set_compute()

    @abstractmethod
    def get_rank(self, process_rank: int) -> int:
        """Retrieve the global rank of the current process.

        Implemented in the inherited class.

        Args:
            process_rank (int): The rank to assign to the process.

        Returns:
            int: The global rank of the process.
        """
        pass

    @abstractmethod
    def get_local_rank(self, process_rank: int) -> int | None:
        """
        Retrieve the local rank of the current process.

        Args:
            process_rank (int): The rank to assign to the process.

        Returns:
            int | None: The local rank. Is None for cpu setups.
        """
        pass

    @abstractmethod
    def get_world_size(self, process_rank: int, nprocs: int) -> int:
        """Retrieve the total number of processes in the distributed training job.

        Implemented in the inherited class.

        Args:
            process_rank (int): The rank to assign to the process.
            nprocs (int): Number of processes to launch.

        Returns:
            int: The total number of processes (world size).
        """
        pass

    @abstractmethod
    def get_master_addr(self) -> str:
        """Return the master node address.

        Implemented in the inherited class.

        Returns:
            str: The master address.
        """
        pass

    @abstractmethod
    def get_master_port(self) -> str:
        """Return the master node port.

        Implemented in the inherited class.

        Returns:
            str: The master port.
        """
        pass

    def _set_cluster_variables(self) -> None:
        """
        Sets the initial default variables for the cluster.

        For now it only supports SLURM Cluster, and should be extended to others
        when needed.
        """
        self.job_id = int(os.environ.get("SLURM_JOB_ID", 93345))
        self.num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        self.node_list = os.environ.get("SLURM_JOB_NODELIST", "Unknown")
        self.node_rank = int(os.environ.get("SLURM_NODEID", 0))
        self.node_name = os.environ.get("SLURMD_NODENAME", "Unknown")
        # currently we only do this for GPUs
        # TODO: extend support to TPUs, CPUs, etc.
        self.cores_per_node: int = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
        )

    def _set_compute(self) -> None:
        """
        Set the compute (cpu or gpu) for the current process based on the compute setup.

        The method checks for CUDA availability and selects the appropriate device.
        If compute_setup is set to "gpu" but CUDA is unavailable, a RuntimeError is raised.

        Raises:
            RuntimeError: If compute_setup is "gpu" but no CUDA devices are available.
        """
        if self.compute_setup == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("Compute setup set to 'gpu' but no CUDA devices are available.")
            self.compute = "gpu"
        elif self.compute_setup == "auto":
            self.compute = "gpu" if torch.cuda.is_available() else "cpu"
        else:
            self.compute = "cpu"

    def set_device(self, local_rank: int | None) -> None:
        """Set the computation device (cpu or cuda:<n>) for the current process based on the compute setup."""
        if self.compute == "gpu":
            self.device = f"cuda:{local_rank}"
            torch.cuda.set_device(local_rank)
        else:
            self.device = "cpu"

        if self.log_setup == "auto":
            self.log_device = self.device
        elif self.log_setup == "cpu":
            self.log_device = "cpu"
        else:
            raise ValueError(f"log_setup {self.log_setup} not supported. Choose 'auto' or 'cpu'.")


# --- DefaultExecution Implementation ---
class DefaultExecution(BaseExecution):
    """
    Default execution for SLURM-like environments.

    Uses SLURM-specific environment variables when available.
    """

    def get_rank(self, process_rank: int) -> int:
        """
        Retrieve the global rank of the current process.

        Args:
            process_rank (int): The rank to assign to the process.

        Returns:
            int: The global rank of the process.
            Priority is given to the "RANK" environment variable; if not found, in a SLURM environment,
            the "SLURM_PROCID" is used. Defaults to 0.
        """
        if self.compute == "cpu":
            return int(process_rank)
        local_rank = self.get_local_rank(process_rank)
        multi_node_index = self.node_rank * self.cores_per_node
        return int(multi_node_index + local_rank) if local_rank else int(multi_node_index)

    def get_local_rank(self, process_rank: int) -> int | None:
        """
        Retrieve the local rank of the current process.

        Args:
            process_rank (int): The rank to assign to the process.

        Returns:
            int | None: The local rank. Uses the "LOCAL_RANK" environment variable if set.
        """
        if self.compute == "cpu":
            return None
        return int(os.environ.get("LOCAL_RANK", process_rank))

    def get_world_size(self, process_rank: int, nprocs: int) -> int:
        """
        Retrieve the total number of processes in the distributed training job.

        Args:
            process_rank (int | None): The rank to assign to the process.
            nprocs (int): Number of processes to launch.

        Returns:
            int: The total number of processes (world size).
                Uses the "WORLD_SIZE" environment variable if set.
        """
        return int(os.environ.get("WORLD_SIZE", nprocs))

    def get_master_addr(self) -> str:
        """
        Determine the master node's address for distributed training.

        Returns:
            str: The master address. If the environment variable "MASTER_ADDR" is set, that value is used.
                 In a SLURM environment, the first hostname from the SLURM node list is used.
                 Defaults to "localhost" if none is found.
        """
        if "MASTER_ADDR" in os.environ:
            return os.environ["MASTER_ADDR"]
        try:
            output = subprocess.check_output(
                ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]]
            )
            return output.splitlines()[0].decode("utf-8").strip()
        except Exception:
            return "localhost"

    def get_master_port(self) -> str:
        """
        Determine the master node's port for distributed training.

        Returns:
            str: The master port. Uses the environment variable "MASTER_PORT" if set.
                 In a SLURM environment, computes a port based on the SLURM_JOB_ID.
                 Defaults to a specific port if not set or on error.
        """
        if "MASTER_PORT" in os.environ:
            return os.environ["MASTER_PORT"]
        try:
            # Compute port based on SLURM_JOB_ID to avoid conflicts.
            return str(int(12000 + int(self.job_id) % 5000))
        except Exception:
            return "12542"


class TorchRunexecution(BaseExecution):
    """
    Execution for torchrun or when using TORCHELASTIC.

    Expects that environment variables like RANK, LOCAL_RANK, WORLD_SIZE,
    MASTER_ADDR, and MASTER_PORT are already set.
    """

    def get_rank(self, process_rank: int) -> int:
        """
        Retrieve the global rank of the current process set by torchrun.

        Args:
            process_rank (int): The rank to assign to the process.

        Returns:
            int: The global rank of the process.
        """
        if self.compute == "cpu":
            return int(process_rank)
        return int(os.environ.get("RANK", process_rank))

    def get_local_rank(self, process_rank: int) -> int | None:
        """
        Retrieve the local rank of the current process (its index on the local node).

        Args:
            process_rank (int): The rank to assign to the process.

        Returns:
            int | None: The local rank. Uses the "LOCAL_RANK" environment variable if set.
        """
        if self.compute == "cpu":
            return None
        return int(os.environ.get("LOCAL_RANK", process_rank))

    def get_world_size(self, process_rank: int, nprocs: int) -> int:
        """
        Retrieve the total number of processes in the distributed training job.

        Args:
            process_rank (int): The rank to assign to the process.
            nprocs (int): Number of processes to launch.

        Returns:
            int: The total number of processes (world size).
                 Uses the "WORLD_SIZE" environment variable if set.
        """
        return int(os.environ.get("WORLD_SIZE", nprocs))

    def get_master_addr(self) -> str:
        """
        Determine the master node's address for distributed training set by torchrun.

        Returns:
            str: The master address.
        """
        return os.environ.get("MASTER_ADDR", "localhost")

    def get_master_port(self) -> str:
        """
        Determine the master node's port for distributed training set by torchrun.

        Returns:
            str: The master port.
        """
        return os.environ.get("MASTER_PORT", "12364")


def detect_execution(
    compute_setup: str = "auto",
    log_setup: str = "cpu",
    dtype: torch.dtype = torch.float32,
    backend: str = "nccl",
) -> tuple[BaseExecution, ExecutionType]:
    """
    Detect and return the appropriate execution instance.

    If no explicit execution is provided, auto-detect using environment variables.

    Args:
        compute_setup (str): Compute device setup; options are "auto" (default), "gpu", or "cpu".
            - "auto": Uses GPU if available, otherwise CPU.
            - "gpu": Forces GPU usage, raising an error if no CUDA device is available.
            - "cpu": Forces CPU usage.
        log_setup (str): Logging device setup; options are "auto", "cpu" (default).
            - "auto": Uses same device to log as used for computation.
            - "cpu": Forces CPU logging.
        dtype (torch.dtype): Data type for controlling numerical precision. Default is torch.float32.
        backend (str): Backend to use for distributed communication (default: "nccl").

    Returns:
        tuple[BaseExecution, ExecutionType]: tuple of
            - Instance of the appropriate execution used for launching the code.
            - Appropriate ExecutionType
    """
    execution = (
        ExecutionType.TORCHRUN if "TORCHELASTIC_RUN_ID" in os.environ else ExecutionType.DEFAULT
    )

    if execution == ExecutionType.TORCHRUN:
        return TorchRunexecution(compute_setup, log_setup, backend, dtype), execution
    else:
        return DefaultExecution(compute_setup, log_setup, backend, dtype), execution
