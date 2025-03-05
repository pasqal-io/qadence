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

from qadence.ml_tools.train_utils.distribution import Distributor
from qadence.ml_tools.data import data_to_device, InfiniteTensorDataset, DictDataLoader
from qadence.types import ExecutionType

logger = getLogger("ml_tools")


class Accelerator(Distributor):
    """
    A class for handling distributed training.

    This class extends `Distributor` to manage distributed training using PyTorch's
    `torch.distributed` API. It supports spawning multiple processes and wrapping models with
    `DistributedDataParallel` (DDP) when required.

    This class is provides head level method - distribute() - which wraps a function at a head process level,
    before launching `nprocs` processes as required. Furthermore, it provides processes level methods,
    such as prepare(), and prepare_batch() which can be run inside each process for correct movement and
    preparation of model, optimizers and datasets.

    Inherited Attributes:
        nprocs (int): Number of processes to launch for distributed training.
        execution (BaseExecution): Detected execution instance for process launch (e.g., "torchrun","default").
        execution_type (ExecutionType): Type of execution used.
        rank (int): Global rank of the process (to be set during environment setup).
        world_size (int): Total number of processes (to be set during environment setup).
        local_rank (int | None): Local rank on the node (to be set during environment setup).
        master_addr (str): Master node address (to be set during environment setup).
        master_port (str): Master node port (to be set during environment setup).
        node_rank (int): Rank of the node on the cluster setup.

    NOTE: There are three different indicators for number of processes executed.
        - 1. self._config_nprocs: Number of processes specified by the user.
        Provided in the initilization of the Accelerator. (acc = Accelerator(nprocs = 2))
        - 2. self.nprocs: Number of processes defined at the head level.
            - When accelerator is used to spawn processes (e.g., In case default, python execution),
            nprocs = _config_nprocs.
            - When an external elastic method is used to spawn processes (e.g., In case of torchrun),
            nprocs = 1. This is because the external launcher already spawns multiple processes,
            and the accelerator __init__ is called from each process.
        - 3. self.world_size: Number of processes actually executed.
    """

    # -----------------------------------------------------------------------------
    # HEAD level methods
    # -----------------------------------------------------------------------------
    def __init__(
        self,
        nprocs: int = 1,
        compute_setup: str = "auto",
        log_setup: str = "cpu",
        backend: str = "gloo",
        dtype: torch_dtype | None = None,
    ) -> None:
        """
        Initializes the Accelerator class.

        Args:
            nprocs (int): Number of processes to launch. Default is 1.
            compute_setup (str): Compute device setup; options are "auto" (default), "gpu", or "cpu".
                - "auto": Uses GPU if available, otherwise CPU.
                - "gpu": Forces GPU usage, raising an error if no CUDA device is available.
                - "cpu": Forces CPU usage.
            log_setup (str): Logging device setup; options are "auto", "cpu" (default).
                - "auto": Uses same device to log as used for computation.
                - "cpu": Forces CPU logging.
            backend (str): The backend for distributed communication. Default is "gloo".
            dtype (torch.dtype | None): Data type for controlling numerical precision. Default is None.
        """
        super().__init__(nprocs, compute_setup, log_setup, backend, dtype)

        # Default values
        self.rank = 0
        self.local_rank = 0
        self.world_size = self.execution.get_world_size(0, self.nprocs)

    def distribute(self, fun: Callable) -> Callable:
        """
        Decorator to distribute the fit function across multiple processes.

        This function is generic and can work with other methods as well.
        Weather it is bound or unbound.

        When applied to a function (typically a fit function), this decorator
        will execute the function in a distributed fashion using torch.multiprocessing.
        The number of processes used is determined by `self.nprocs`,
        and if multiple nodes are involved (`self.num_nodes > 1`), the process count is
        adjusted accordingly. In single process mode (`self.nporcs` is 1), the function
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
        def wrapper(*args: Any, **kwargs: Any) -> Any:

            # Get the original picklable function
            # for the case of bound class method
            # as well as a function
            if self.is_class_method(fun, args):
                instance = args[0]
                method_name = fun.__name__
                method = getattr(instance, method_name)
                args = args[1:]
                self._spawn_method(instance, method, args, kwargs)
            else:
                instance = None
                # method_name = fun.__name__
                # module = inspect.getmodule(fun)
                # method = getattr(module, method_name) if module else fun
                self._spawn_method(instance, fun, args, kwargs)

            if instance and hasattr(instance, "accelerator"):
                instance.accelerator.finalize()
            else:
                self.finalize()

            # TODO: Return the original returns from fun
            # Currently it only returns the model and optimizer
            # similar to the fit method.
            try:
                return instance.model, instance.optimizer
            except Exception:
                return

        return wrapper

    def worker(self, rank: int, instance: Any, fun: Callable, args: tuple, kwargs: dict) -> None:
        """
        Worker function to be executed in each spawned process.

        This function is called in every subprocess created by torch.multiprocessing (via mp.spawn).
        It performs the following tasks:
          1. Sets up the accelerator for the given process rank. This typically involves configuring
             the GPU or other hardware resources for distributed training.
          2. If the retrieved method has been decorated (i.e. it has a '__wrapped__' attribute),
             the original, unwrapped function is invoked with the given arguments. Otherwise,
             the method is called directly.

        Args:
            rank (int): The rank (or identifier) of the spawned process.
            instance (object): The object (Trainer) that contains the method to execute.
                               This object is expected to have an `accelerator` attribute with a `setup_process(rank)` method.
                               This argument is optional, in case it is None, the fun will be called independently.
            fun (Callable): The function of the method on the instance to be executed.
            args (tuple): Positional arguments to pass to the target method.
            kwargs (dict): Keyword arguments to pass to the target method.
        """
        # Setup the accelerator for the given process rank (e.g., configuring GPU)
        if instance and instance.accelerator:
            instance.accelerator.setup_process(rank)
        else:
            self.setup_process(rank)

        if hasattr(fun, "__wrapped__"):
            # Explicitly get the original (unbound) method, passing in the instance.
            # We need to call the original method in case so that MP spawn does not
            # create multiple processes. (To Avoid infinite loop)
            fun = fun.__wrapped__  # Unwrap if decorated
            fun(instance, *args, **kwargs) if instance else fun(*args, **kwargs)
        else:
            fun(*args, **kwargs)

    def is_class_method(self, fun: Callable, args: Any) -> bool:
        """
        Determines if `fun` is a class method or a standalone function.

        Frist argument of the args should be:
        - An object and has __dict__: making it a class
        - Has a method named fun: making it a class that has this method.

        Args:
            fun (Callable): The function being checked.
            args (tuple): The arguments passed to the function.

        Returns:
            bool: True if `fun` is a class method, False otherwise.
        """
        return (
            bool(args)
            and isinstance(args[0], object)
            and hasattr(args[0], "__dict__")
            and hasattr(args[0], fun.__name__)
        )

    def _spawn_method(self, instance: Any, method: Callable, args: Any, kwargs: Any) -> None:
        """
        This method spawns the required numbers of processes.

        - if execution is `default`, it will spawn `nproc` processes across all nodes
        - if execution is `otherwise`, it will run a single process.

        Args:
            instance (object): The object (Trainer) that contains the method to execute.
                               This object is expected to have an `accelerator` attribute with a `setup_process(rank)` method.
                               This argument is optional, in case it is None, the fun will be called independently.
            method (Callable): The function of the method on the instance to be executed.
            args (tuple): Positional arguments to pass to the target method.
            kwargs (dict): Keyword arguments to pass to the target method.
        """

        if self.execution_type == ExecutionType.DEFAULT and self.world_size > 1:
            # Spawn multiple processes that will run the worker function.
            nprocs = self.nprocs
            if self.execution.num_nodes > 1:
                nprocs //= self.execution.num_nodes
            mp.spawn(
                self.worker,
                args=(instance, method, args, kwargs),
                nprocs=int(nprocs),
                join=True,
            )
        else:
            # In single process mode, call the worker with rank 0.
            self.worker(0, instance, method, args, kwargs)

    # -----------------------------------------------------------------------------
    # PROCESS level methods
    # -----------------------------------------------------------------------------
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
        model = model.to(device=self.execution.device, dtype=self.execution.dtype)

        # If using distributed training with more than one device:
        if self.world_size > 1:
            if self.execution.device.startswith("cuda"):
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
        if batch is None:
            return None

        if isinstance(batch, dict):
            return {
                key: data_to_device(
                    value, device=self.execution.device, dtype=self.execution.data_dtype
                )
                for key, value in batch.items()
            }
        elif isinstance(batch, (tuple, list)):
            return tuple(
                data_to_device(x, device=self.execution.device, dtype=self.execution.data_dtype)
                for x in batch
            )
        elif isinstance(batch, torch.Tensor):
            return data_to_device(
                batch, device=self.execution.device, dtype=self.execution.data_dtype
            )
        return

    def all_reduce_dict(
        self, d: dict[str, torch.Tensor], op: str = "mean"
    ) -> dict[str, torch.Tensor]:
        """
        Performs an all-reduce operation on a dictionary of tensors, averaging values across all processes.

        Args:
            d (dict[str, torch.Tensor]): A dictionary where values are tensors to be reduced across processes.
            op (str): Operation method to all_reduce with. Available options include `sum`, `avg`, and `max`.
                            Defaults to `avg`

        Returns:
            dict[str, torch.Tensor]: A dictionary with the reduced tensors, averaged over the world size.
        """
        if dist.is_initialized():
            world_size = dist.get_world_size()
            reduced: dict[str, torch.Tensor] = {}
            for key, tensor in d.items():
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(
                        tensor, device=self.execution.device, dtype=self.execution.data_dtype
                    )
                tensor = tensor.detach().clone()
                if op == "max":
                    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                elif op == "sum":
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                else:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor /= world_size
                reduced[key] = tensor
            return reduced
        else:
            return d

    def broadcast(self, obj: Any, src: int) -> Any:
        """
        Broadcasts an object from the source process to all processes.

        On non-source processes, this value is ignored.

        Args:
            obj (Any): The object to broadcast on the source process.
            src (int): The source process rank.

        Returns:
            Any : The broadcasted object from the source process.
        """
        if dist.is_initialized():
            obj_list = [obj] if self.rank == src else [None]
            dist.broadcast_object_list(obj_list, src=src)
            return obj_list[0]
        else:
            return obj
