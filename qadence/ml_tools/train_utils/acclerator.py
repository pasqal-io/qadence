from __future__ import annotations
from logging import getLogger
from typing import Any, Callable
import functools

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import dtype as torch_dtype
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from qadence.ml_tools.train_utils.strategy import DistributionStrategy
from qadence.ml_tools.data import data_to_device, InfiniteTensorDataset, DictDataLoader

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
        nprocs: int = 1,
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

    def setup(self, process_rank: int) -> None:
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
        logger.info("  Node             : %s", str(self.node_name))
        logger.info("  Rank             : %s", str(self.rank))
        logger.info("  Local Rank       : %s", str(self.local_rank))
        logger.info("  World Size       : %s", str(self.world_size))
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

        This method iterates over the provided objects and:
        - Moves models to the specified device (e.g., GPU or CPU) and casts them to the
            desired precision (specified by `self.dtype`). It then wraps models in
            DistributedDataParallel (DDP) if more than one device is used.
        - Passes through optimizers unchanged.
        - For dataloaders, it adjusts them to use a distributed sampler (if applicable)
            by calling a helper method. Note that only the sampler is prepared; moving the
            actual batch data to the device is handled separately during training.
            Please use the `prepare_batch` method to move the batch to correct device/dtype.

        Args:
            *args (Any): A variable number of objects to be prepared. These can include:
                - PyTorch models (`nn.Module`)
                - Optimizers (`optim.Optimizer`)
                - DataLoaders (or a dictionary-like `DictDataLoader` of dataloaders)

        Returns:
            tuple[Any, ...]: A tuple containing the prepared objects, where each object has been
                            modified as needed to support distributed training.
        """
        prepared: list = []
        for obj in args:
            if obj is None:
                prepared.append(None)
            elif isinstance(obj, nn.Module):
                prepared.append(self._prepare_model(obj))
            elif isinstance(obj, optim.Optimizer):
                prepared.append(self._prepare_optimizer(obj))
            elif isinstance(obj, (DataLoader, DictDataLoader)):
                prepared.append(self._prepare_data(obj))
            else:
                prepared.append(obj)
        return tuple(prepared)

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        """
        Moves the model to the desired device and casts it to the specified dtype.

        In a distributed setting, if more than one device is used (i.e., self.world_size > 1),
        the model is wrapped in DistributedDataParallel (DDP) to handle gradient synchronization
        across devices.

        Args:
            model (nn.Module): The PyTorch model to prepare.

        Returns:
            nn.Module: The model moved to the correct device (and wrapped in DDP if applicable).
        """
        model = model.to(device=self.device, dtype=self.dtype)

        # If using distributed training with more than one device:
        if self.world_size > 1:
            if self.device.startswith("cuda"):
                # For GPU-based training: wrap the model with DDP and specify the local GPU.
                model = DDP(model, device_ids=[self.local_rank])
            else:
                # For CPU-based or other environments:
                if not self.local_rank:
                    model = DDP(model)

        return model

    def _prepare_optimizer(self, optimizer: optim.Optimizer) -> optim.Optimizer:
        """
        Passes through the optimizer without modification.

        In this preparation routine, optimizers do not require moving to a specific device or
        changing precision. They are simply returned as provided.

        Args:
            optimizer (optim.Optimizer): The optimizer to prepare.

        Returns:
            optim.Optimizer: The unmodified optimizer.
        """
        # Optimizers are not device-specific in this context, so no action is needed.
        return optimizer

    def _prepare_data(self, dataloader: DataLoader | DictDataLoader) -> DataLoader | DictDataLoader:
        """
        Adjusts DataLoader(s) for distributed training.

        For a single DataLoader, this method applies the necessary adjustments (e.g., setting up a
        distributed sampler). If a DictDataLoader (a container for multiple DataLoaders) is provided,
        each contained DataLoader is prepared individually.

        Args:
            dataloader (Union[DataLoader, DictDataLoader]): The dataloader or dictionary of dataloaders to prepare.

        Returns:
            Union[DataLoader, DictDataLoader]: The prepared dataloader(s) with the correct distributed
                                            sampling setup.
        """
        if isinstance(dataloader, DictDataLoader):
            # If the input is a DictDataLoader, prepare each contained DataLoader.
            prepared_dataloaders = {
                key: self._prepare_dataloader(dl) for key, dl in dataloader.dataloaders.items()
            }
            return DictDataLoader(prepared_dataloaders)
        else:
            # For a single DataLoader, prepare it directly.
            return self._prepare_dataloader(dataloader)

    def _prepare_dataloader(self, dataloader: DataLoader) -> DataLoader:
        """
        Prepares a single DataLoader for distributed training.

        When training in a distributed setting (i.e., when `self.world_size > 1`), data must be
        divided among multiple processes. This is achieved by creating a
        DistributedSampler that splits the dataset into distinct subsets for each process.

        This method does the following:
        - If distributed training is enabled:
            - Checks if the dataset is not an instance of `InfiniteTensorDataset`.
                - If so, creates a `DistributedSampler` for the dataset using the total number
                    of replicas (`self.world_size`) and the current process's rank (`self.local_rank`).
                - Otherwise (i.e., for infinite datasets), no sampler is set (sampler remains `None`).
            - Returns a new DataLoader configured with:
                - The same dataset and batch size as the original.
                - The distributed sampler (if applicable).
                - The number of workers and pin_memory settings retrieved from the original DataLoader.
        - If not in a distributed setting (i.e., `self.world_size <= 1`), returns the original DataLoader unmodified.

        Args:
            dataloader (DataLoader): The original DataLoader instance that loads the dataset.

        Returns:
            DataLoader: A new DataLoader prepared for distributed training if in a multi-process environment;
                        otherwise, the original DataLoader is returned.
        """
        if self.world_size > 1:
            if not isinstance(dataloader.dataset, InfiniteTensorDataset):
                # If the dataset is not an infinite dataset, create a DistributedSampler.
                sampler = DistributedSampler(
                    dataloader.dataset, num_replicas=self.world_size, rank=self.local_rank
                )
            else:
                # For infinite datasets, we do not use a sampler since the dataset
                # is designed to loop indefinitely.
                sampler = None

            return DataLoader(
                dataloader.dataset,  # Use the same dataset as the original.
                batch_size=dataloader.batch_size,  # Maintain the same batch size.
                sampler=sampler,  # Use the created DistributedSampler (or None).
                num_workers=getattr(dataloader, "num_workers", 0),
                pin_memory=getattr(dataloader, "pin_memory", False),
            )
        return dataloader

    def prepare_batch(self, batch: dict | list | tuple | torch.Tensor | None) -> Any:
        """
        Moves a batch of data to the target device and casts it to the desired data dtype.

        This method is typically called within the optimization step of your training loop.
        It supports various batch formats:
            - If the batch is a dictionary, each value is moved individually.
            - If the batch is a tuple or list, each element is processed and returned as a tuple.
            - Otherwise, the batch is processed directly.

        Args:
            batch (Any): The batch of data to move to the device. This can be a dict, tuple, list,
                         or any type compatible with `data_to_device`.

        Returns:
            Any: The batch with all elements moved to `self.device` and cast to `self.data_dtype`.
        """
        if batch:
            if isinstance(batch, dict):
                return {
                    key: data_to_device(value, device=self.device, dtype=self.data_dtype)
                    for key, value in batch.items()
                }
            elif isinstance(batch, (tuple, list)):
                return tuple(
                    data_to_device(x, device=self.device, dtype=self.data_dtype) for x in batch
                )
            else:
                return data_to_device(batch, device=self.device, dtype=self.data_dtype)

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

    def worker(self, rank: int, instance: Any, method_name: str, args: tuple, kwargs: dict) -> None:
        """
        Worker function to be executed in each spawned process.

        This function is called in every subprocess created by torch.multiprocessing (via mp.spawn).
        It performs the following tasks:
          1. Sets up the accelerator for the given process rank. This typically involves configuring
             the GPU or other hardware resources for distributed training.
          2. Retrieves the method specified by `method_name` from the provided `instance`.
          3. If the retrieved method has been decorated (i.e. it has a '__wrapped__' attribute),
             the original, unwrapped function is invoked with the given arguments. Otherwise,
             the method is called directly.

        Args:
            rank (int): The rank (or identifier) of the spawned process.
            instance (object): The object (Trainer) that contains the method to execute.
                               This object is expected to have an `accelerator` attribute with a `setup(rank)` method.
            method_name (str): The name of the method on the instance to be executed.
            args (tuple): Positional arguments to pass to the target method.
            kwargs (dict): Keyword arguments to pass to the target method.
        """
        # Setup the accelerator for the given process rank (e.g., configuring GPU)
        instance.accelerator.setup(rank)
        method = getattr(instance, method_name)

        # If the method is wrapped by a decorator, retrieve the original function.
        if hasattr(method, "__wrapped__"):
            # Explicitly call the original (unbound) method, passing in the instance.
            # We need to call the original method in case so that MP spawn does not
            # create multiple processes.
            original_method = method.__wrapped__
            original_method(instance, *args, **kwargs)
        else:
            # Otherwise, simply call the method.
            method(*args, **kwargs)

    def distribute(self, fun: Callable) -> Callable:
        """
        Decorator to distribute the fit function across multiple processes.

        This function is
        generic and can work with other methods as well. Weather it is bound or unbound.

        When applied to a function (typically a fit function), this decorator
        will execute the function in a distributed fashion using torch.multiprocessing if
        `self.spawn` is True. The number of processes used is determined by `self.nprocs`,
        and if multiple nodes are involved (`self.num_nodes > 1`), the process count is
        adjusted accordingly. In single process mode (`self.spawn` is False), the function
        is executed directly in the current process.

        After execution, the decorator returns the model stored in `instance.model`.

        Parameters:
            fun (callable): The function to be decorated. This function usually implements
                            a model fitting or training routine.

        Returns:
            callable: The wrapped function. When called, it will execute in distributed mode
                      (if configured) and return the value of `instance.model`.
        """

        @functools.wraps(fun)
        def wrapper(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if self.spawn:
                # Spawn multiple processes that will run the worker function.
                nprocs = self.nprocs
                if self.num_nodes > 1:
                    nprocs //= self.num_nodes
                mp.spawn(
                    self.worker,
                    args=(instance, fun.__name__, args, kwargs),
                    nprocs=int(nprocs),
                    join=True,
                )
            else:
                # In single process mode, call the worker with rank 0.
                self.worker(0, instance, fun.__name__, args, kwargs)

            # TODO: Return the original returns from fun
            # Currently it only returns the model and optimizer
            # similar to the fit method.
            try:
                return instance.model, instance.optimizer
            except Exception:
                return

        return wrapper
