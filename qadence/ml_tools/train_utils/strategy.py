from __future__ import annotations


from logging import getLogger

import os
import subprocess
import torch
import torch.distributed as dist

# Initialize the logger for this module
logger = getLogger("ml_tools")


class DistributionStrategy:
    """
    Class to set up and manage distributed training environments.

    This class detects the current launch strategy (e.g., torchrun, SLURM, or default),
    configures environment variables required for distributed training (such as rank, world size,
    master address, and master port), and sets the appropriate computation device (CPU or GPU).
    It also provides methods to start and clean up the PyTorch distributed process group.

    Attributes:
        spawn (bool): Whether to use multiprocessing spawn mode for process initialization.
        nprocs (int): Number of processes to launch for distributed training.
        strategy (str): Detected strategy for process launch ("torchrun", "slurm", or "default").
        backend (str): The backend used for distributed communication (e.g., "nccl", "gloo").
            It should be one of the backends supported by torch.distributed
        compute_setup (str): Desired computation device setup.
        log_setup (str): Desired logging device setup.
        rank (int | None): Global rank of the process (to be set during environment setup).
        world_size (int | None): Total number of processes (to be set during environment setup).
        local_rank (int | None): Local rank on the node (to be set during environment setup).
        master_addr (str | None): Master node address (to be set during environment setup).
        master_port (str | None): Master node port (to be set during environment setup).
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
        dtype: torch.dtype | None = torch.float32,
        backend: str = "nccl",
    ) -> None:
        """
        Initialize the DistributionStrategy.

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
        """
        self.backend: str = backend
        self.compute_setup: str = compute_setup
        self.log_setup: str = log_setup
        self.strategy: str
        self.rank: int
        self.world_size: int
        self.local_rank: int | None
        self.master_addr: str
        self.master_port: str
        self.compute: str
        self.device: str
        self.log_device: str
        self.spawn: bool = False
        self._nprocs: int = 1
        self.dtype: torch.dtype | None = dtype
        self.data_dtype: torch.dtype | None = None
        # TODO: This is legacy behavior of data_dtype. Modify it appropriately.
        if self.dtype:
            self.data_dtype = torch.float64 if (self.dtype == torch.complex128) else self.dtype
        self._set_cluster_variables()

    @property
    def nprocs(self) -> int:
        """
        Returns the test nprocs for the accelerator.

        Returns:
            int: Number of processes.
        """
        return self._nprocs

    @nprocs.setter
    def nprocs(self, nprocs: int) -> None:
        """
        Sets the number of processes and the spawn variable.

        Args:
            nprocs (int): Number of processes
        """
        self.spawn = nprocs > 1
        self._nprocs = nprocs

    def detect_strategy(self) -> str:
        """
        Detect the launch strategy based on environment variables.

        Returns:
            str: The detected launch strategy. Possible values are:
                - "Default": If the "LOCAL_RANK" environment variable is set.
                    Possibly by torchrun or my spawned processes.
                - "slurm": If the "SLURM_PROCID" environment variable is set.
                - "none": Otherwise.
        """
        if ("LOCAL_RANK" in os.environ) or self.spawn:
            return "default"
        elif "TORCHELASTIC_RUN_ID" in os.environ:
            return "torchrun"
        else:
            return "none"

    def _set_cluster_variables(self) -> None:
        """
        Sets the initial default variables for the cluster.

        For now it only supports SLURM Cluster, and should be extended to others
        when needed.
        """
        self.job_id = int(os.environ.get("SLURM_JOB_ID", 93345))
        self.num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
        self.node_list = os.environ.get("SLURM_JOB_NODELIST", "Unknown")
        # currently we only do this for GPUs
        # TODO: extend support to TPUs, CPUs, etc.
        self.cores_per_node: int = (
            int(torch.cuda.device_count()) if torch.cuda.is_available() else 1
        )

    def setup_distributed_environment(self, process_rank: int) -> dict[str, int | None]:
        """
        Set up environment variables and the computation device for distributed processing.

        This method retrieves the global rank, world size, and local rank using helper methods,
        sets the corresponding environment variables, and if running in a multi-process setting,
        sets up the master address and port for distributed communication. Finally, it configures
        the computation device based on the specified compute setup.
        This method sets:
            rank (int | None): Global rank of the process (to be set during environment setup).
            world_size (int | None): Total number of processes (to be set during environment setup).
            local_rank (int | None): Local rank on the node (to be set during environment setup).
            master_addr (str | None): Master node address (to be set during environment setup).
            master_port (str | None): Master node port (to be set during environment setup).
            node_rank (int): Rank of the node on the cluster setup.
            node_name (str): Name of the node on the cluster setup.

        Args:
            process_rank (int | None): The rank to assign to the process (used in spawn scenarios).

        Returns:
            dict[str, int | None]: A dictionary containing the global rank, world size, and local rank.
        """
        # set the process based variables
        self.node_rank = int(os.environ.get("SLURM_NODEID", 0))
        self.node_name = os.environ.get("SLURMD_NODENAME", "Unknown")

        self._set_compute()
        self.local_rank = self._get_local_rank(process_rank)
        self.world_size = self._get_world_size(process_rank)
        self.rank = self._get_rank(process_rank)
        self._set_device()
        # Set environment variables for distributed training
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["MASTER_ADDR"] = self.master_addr = self._get_master_addr()
        os.environ["MASTER_PORT"] = self.master_port = self._get_master_port()

        return {"RANK": self.rank, "WORLD_SIZE": self.world_size, "LOCAL_RANK": self.local_rank}

    def _get_master_addr(self) -> str:
        """
        Determine the master node's address for distributed training.

        Returns:
            str: The master address. If the environment variable "MASTER_ADDR" is set, that value is used.
                 In a SLURM environment, the first hostname from the SLURM node list is used.
                 Defaults to "localhost" if none is found.
        """
        if "MASTER_ADDR" in os.environ:
            return os.environ["MASTER_ADDR"]
        elif self.strategy == "default":
            try:
                # Use scontrol to get hostnames from SLURM's node list and select the first one
                return (
                    subprocess.check_output(
                        ["scontrol", "show", "hostnames", os.environ["SLURM_NODELIST"]]
                    )
                    .splitlines()[0]
                    .decode("utf-8")
                    .strip()
                )
            except Exception:
                return "localhost"
        else:
            return "localhost"

    def _get_master_port(self) -> str:
        """
        Determine the master node's port for distributed training.

        Returns:
            str: The master port. Uses the environment variable "MASTER_PORT" if set.
                 In a SLURM environment, computes a port based on the SLURM_JOB_ID.
                 Defaults to a specific port if not set or on error.
        """
        if "MASTER_PORT" in os.environ:
            return os.environ["MASTER_PORT"]
        elif self.strategy == "default":
            try:
                # Calculate the port number based on SLURM_JOB_ID to avoid conflicts
                return str(int(12000 + int(self.job_id) % 5000))
            except Exception:
                return "12542"
        else:
            return "12364"

    def _get_rank(self, process_rank: int) -> int:
        """
        Retrieve the global rank of the current process.

        Args:
            process_rank (int | None): The rank to assign to the process (used in spawn scenarios).

        Returns:
            int: The global rank of the process.
                 Priority is given to the "RANK" environment variable; if not found, in a SLURM environment,
                 the "SLURM_PROCID" is used. Defaults to 0.
        """
        if self.compute == "cpu":
            return int(process_rank)
        if "RANK" in os.environ:
            return int(os.environ["RANK"])
        if self.strategy == "default":
            rank = self.node_rank * self.cores_per_node
            if self.local_rank:
                rank += self.local_rank
            return int(rank)
        return 0

    def _get_world_size(self, process_rank: int) -> int:
        """
        Retrieve the total number of processes in the distributed training job.

        Args:
            process_rank (int | None): The rank to assign to the process (used in spawn scenarios).

        Returns:
            int: The total number of processes (world size).
                 Uses the "WORLD_SIZE" environment variable if set, or "SLURM_NTASKS" in a SLURM environment.
                 Defaults to 1.
        """
        if "WORLD_SIZE" in os.environ:
            return int(os.environ["WORLD_SIZE"])
        if self.strategy == "default":
            return int(self.nprocs)
        return 1

    def _get_local_rank(self, process_rank: int) -> int | None:
        """
        Retrieve the local rank of the current process (its index on the local node).

        Args:
            process_rank (int | None): The rank to assign to the process (used in spawn scenarios).

        Returns:
            int: The local rank.
                 Uses the "LOCAL_RANK" environment variable if set, or "SLURM_LOCALID" in a SLURM environment.
                 Defaults to 0.
        """
        if self.compute == "cpu":
            # No local rank in case of CPUs for true parallelism
            return None
        if "LOCAL_RANK" in os.environ:
            return int(os.environ["LOCAL_RANK"])
        if self.strategy == "default":
            return int(process_rank)
        return 0

    def _set_compute(self) -> None:
        """
        Set the compute (cpu or gpu) for the current process based on the compute setup.

        The method checks for CUDA availability and selects the appropriate device.
        If compute_setup is set to "gpu" but CUDA is unavailable, a RuntimeError is raised.

        Raises:
            RuntimeError: If compute_setup is "gpu" but no CUDA devices are available.
        """
        compute = "cpu"
        if self.compute_setup == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"Compute setup set to {self.compute_setup} but no CUDA devices are available."
                )
            else:
                compute = "gpu"
        if self.compute_setup == "auto":
            if torch.cuda.is_available():
                compute = "gpu"
        self.compute = compute

    def _set_device(self) -> None:
        """Set the computation device (cpu or cuda:<n>) for the current process based on the compute setup."""
        if self.compute == "gpu":
            self.device = f"cuda:{self.local_rank}"
            torch.cuda.set_device(self.local_rank)
        else:
            self.device = "cpu"

        if self.log_setup == "auto":
            self.log_device = self.device
        elif self.log_setup == "cpu":
            self.log_device = "cpu"
        else:
            raise ValueError(
                f"log_setup {self.log_setup} not supported. Choose between `auto` and `cpu`"
            )

    def setup_process(
        self, process_rank: int, nprocs: int
    ) -> tuple[int | None, int | None, int | None, str | None]:
        """
        Set up the process for distributed training, especially useful when processes are spawned.

        This method optionally sets environment variables for a spawned process if a process rank is provided.
        It then calls setup_distributed_environment() to configure the global settings and logs a warning if the provided
        number of processes does not match the environment's world size (only for the master process).

        Args:
            process_rank (int | None): The rank to assign to the process (used in spawn scenarios).
            nprocs (int | None): The total number of processes expected. Used for validation against the environment.

        Returns:
            tuple[int, int, int, str]: A tuple containing:
                - rank (int): Global rank of the process.
                - world_size (int): Total number of processes.
                - local_rank (int): Local rank on the node.
                - device (str): The computation device (e.g., "cpu" or "cuda:<local_rank>").

        Note:
            The attributes `self.spawn` and `self.nprocs` are referenced here but are assumed to be defined externally.
        """
        self.setup_distributed_environment(process_rank)
        if self.spawn and nprocs and self.rank == 0 and str(nprocs) != str(self.world_size):
            logger.warning(
                "Provided nprocs (%d) does not match environment world size (%d). Using environment world size.",
                nprocs,
                self.world_size,
            )
        return self.rank, self.world_size, self.local_rank, self.device

    def start_process_group(self) -> None:
        """
        Initialize the PyTorch distributed process group for multi-process training.

        If the world size is greater than 1, this method initializes the process group with the specified
        backend, rank, and world size. For the master process (rank 0), it logs configuration details such
        as the total number of nodes, processes, master address, and master port. Finally, it synchronizes
        all processes with a barrier.
        """
        if self.world_size and self.world_size > 1:
            dist.init_process_group(
                backend=self.backend, rank=self.rank, world_size=self.world_size
            )
            if self.rank == 0:
                logger.info("Starting Distributed Process Group")
                logger.info("=============================")
                logger.info("  Total Nodes       : %d", int(os.environ.get("SLURM_NNODES", 1)))
                logger.info("  Total Processes   : %d", self.world_size)
                logger.info("  Master Address    : %s", self.master_addr)
                logger.info("  Master Port       : %s", self.master_port)
            dist.barrier()

    def cleanup_process_group(self) -> None:
        """
        Clean up the PyTorch distributed process group after training is complete.

        If the distributed process group has been initialized, it is destroyed. Additionally, the master
        process (rank 0) logs that the process group is being killed.
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            if self.rank == 0:
                logger.info("Killing Distributed Process Group")
