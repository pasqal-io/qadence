from __future__ import annotations


from logging import getLogger

import os
import torch
import torch.distributed as dist
from qadence.ml_tools.train_utils.execution import BaseExecution, detect_execution
from qadence.types import ExecutionType

# Initialize the logger for this module
logger = getLogger("ml_tools")


class Distributor:
    """
    Class to set up and manage distributed training.

    This class uses the detect_execution() method to get the correct current launch execution
    (e.g., torchrun, default). It provides methods to setup processes, start, and clean up the
    PyTorch distributed process group.

    The execution configures environment variables required for distributed training (such as rank, world size,
    master address, and master port), and sets the appropriate computation device (CPU or GPU).

    Attributes:
        nprocs (int): Number of processes to launch for distributed training.
        execution (BaseExecution): Detected execution instance for process launch (e.g., "torchrun","default").
        execution_type (ExecutionType): Type of exeuction used.
        rank (int): Global rank of the process (to be set during environment setup).
        world_size (int): Total number of processes (to be set during environment setup).
        local_rank (int | None): Local rank on the node (to be set during environment setup).
        master_addr (str): Master node address (to be set during environment setup).
        master_port (str): Master node port (to be set during environment setup).
        node_rank (int): Rank of the node on the cluster setup.
    """

    # -----------------------------------------------------------------------------
    # HEAD level methods
    # -----------------------------------------------------------------------------
    def __init__(
        self,
        nprocs: int,
        compute_setup: str,
        log_setup: str,
        backend: str,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Initialize the Distributor.

        Args:
            nprocs (int): Number of processes to launch.
            compute_setup (str): Compute device setup; options are "auto" (default), "gpu", or "cpu".
                - "auto": Uses GPU if available, otherwise CPU.
                - "gpu": Forces GPU usage, raising an error if no CUDA device is available.
                - "cpu": Forces CPU usage.
            log_setup (str): Logging device setup; options are "auto", "cpu" (default).
                - "auto": Uses same device to log as used for computation.
                - "cpu": Forces CPU logging.
            backend (str): Backend to use for distributed communication (default: "nccl").
            dtype (torch.dtype | None): Data type for controlling numerical precision. Default is None.
        """
        self._nprocs: int
        self.rank: int
        self.world_size: int
        self.local_rank: int | None
        self.master_addr: str
        self.master_port: str
        self.execution: BaseExecution

        self.execution, self.execution_type = detect_execution(
            compute_setup, log_setup, backend, dtype
        )

        self._config_nprocs = nprocs
        if self.execution_type == ExecutionType.TORCHRUN:
            # torchrun already spawns multiple process with required env variables
            self.nprocs = 1
        else:
            self.nprocs = nprocs

    # -----------------------------------------------------------------------------
    # PROCESS level methods
    # -----------------------------------------------------------------------------
    def setup_process(self, process_rank: int) -> None:
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
        self.setup_process_rank_environment(process_rank)

        logger.info("Initializing Accelerator")
        logger.info("=============================")
        logger.info(
            " Node, Device                : %s, %s",
            str(self.execution.node_name),
            self.execution.device,
        )
        logger.info(
            " Rank, Local Rank, World Size: %s, %s, %s",
            str(self.rank),
            str(self.local_rank),
            str(self.world_size),
        )
        logger.info(" Master Address, Master Port : %s, %s", self.master_addr, self.master_port)

        self.start_process_group()
        if self.rank == 0:
            self._log_warnings()  # log the warnings only from the main process

    def setup_process_rank_environment(self, process_rank: int) -> dict[str, int | None]:
        """
        Set up the process for distributed training, especially useful when processes are spawned.

        Set up environment variables and the computation device for distributed processing.

        This method optionally sets environment variables for a spawned process if a process rank is provided.
        This method retrieves the global rank, world size, and local rank using helper methods,
        sets the corresponding environment variables, and if running in a multi-process setting,
        sets up the master address and port for distributed communication. Finally, it configures
        the computation device based on the specified compute setup.
        This method sets:
            rank (int): Global rank of the process (to be set during environment setup).
            world_size (int): Total number of processes (to be set during environment setup).
            local_rank (int | None): Local rank on the node (to be set during environment setup).
            master_addr (str): Master node address (to be set during environment setup).
            master_port (str): Master node port (to be set during environment setup).
            node_rank (int): Rank of the node on the cluster setup.
            node_name (str): Name of the node on the cluster setup.

        Args:
            process_rank (int | None): The rank to assign to the process (used in spawn scenarios).

        Returns:
            dict[str, int | None]: A dictionary containing the global rank, world size, and local rank.
        """
        # set the process based variables
        self.local_rank = self.execution.get_local_rank(process_rank)
        self.world_size = self.execution.get_world_size(process_rank, self.nprocs)
        self.rank = self.execution.get_rank(process_rank)
        self.execution.set_device(self.local_rank)
        # Set environment variables for distributed training
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["LOCAL_RANK"] = str(self.local_rank)
        os.environ["MASTER_ADDR"] = self.master_addr = self.execution.get_master_addr()
        os.environ["MASTER_PORT"] = self.master_port = self.execution.get_master_port()

        return {"RANK": self.rank, "WORLD_SIZE": self.world_size, "LOCAL_RANK": self.local_rank}

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
                backend=self.execution.backend, rank=self.rank, world_size=self.world_size
            )
            if self.rank == 0:
                logger.info("Starting Distributed Process Group")
                logger.info("=============================")
                logger.info(" Total Nodes       : %d", int(os.environ.get("SLURM_NNODES", 1)))
                logger.info(" Total Processes   : %d", self.world_size)
                logger.info(" Master Address    : %s", self.master_addr)
                logger.info(" Master Port       : %s", self.master_port)
            dist.barrier()

    def finalize(self) -> None:
        """
        Clean up the PyTorch distributed process group after training is complete.

        If the distributed process group has been initialized, it is destroyed. Additionally, the master
        process (rank 0) logs that the process group is being killed.
        """
        if dist.is_initialized():
            dist.destroy_process_group()
            if self.rank == 0:
                logger.info("Killing Distributed Process Group")

    def _log_warnings(self) -> None:

        if self.execution_type == ExecutionType.TORCHRUN:
            logger.info(
                f"Process was launched using `torchrun`, "
                "processes spawned will be set based on `torchrun` setup."
            )
        logger.info(f"User sepcifed `nprocs`={self._config_nprocs}")
        logger.info(f"Total processes spawned={self.world_size}")
