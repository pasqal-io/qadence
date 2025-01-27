import torch
import torch.distributed as dist

class DeviceManager:
    def __init__(self, use_gpu=True, distributed=False, world_size=1, rank=0, backend='nccl'):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.backend = backend
        self.device = self._set_device()

    def _set_device(self):
        if self.use_gpu:
            if self.distributed:
                torch.cuda.set_device(self.rank)
                return torch.device(f"cuda:{self.rank}")
            return torch.device("cuda")
        return torch.device("cpu")

    def initialize_training(self, seed=42):
        """Initializes the training environment, sets seed, and initializes distributed backend if applicable."""
        torch.manual_seed(seed)
        if self.use_gpu:
            torch.cuda.manual_seed_all(seed)
        if self.distributed:
            dist.init_process_group(backend=self.backend, world_size=self.world_size, rank=self.rank)
            torch.cuda.set_device(self.rank)
        print(f"Training initialized on device: {self.device}")

    def end_training(self):
        """Ends the training process by cleaning up distributed resources."""
        if self.distributed and dist.is_initialized():
            dist.destroy_process_group()
        print("Training ended and resources cleaned up.")
    
    def transfer_model_to_device(self, model):
        return model.to(self.device)
    
    def transfer_tensor_to_device(self, tensor):
        return tensor.to(self.device)

    def synchronize_gradients(self, model):
        if self.distributed:
            for param in model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad.data)
    
    def get_device(self):
        return self.device
