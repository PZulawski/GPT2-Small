import os 
import tiktoken
import torch
import datetime
from torch.nn import functional as F
import torch.distributed as dist

from config_local import WANDB_API_KEY

def setup(rank, world_size, DDP):
    if DDP:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        acc = torch.accelerator.current_accelerator()
        backend = torch.distributed.get_default_backend_for_device(acc)
        # initialize the process group
        dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup(DDP: bool):
    if DDP:
        dist.destroy_process_group()


def get_tokenizer(name: str = 'tiktoken'):
    if name == 'tiktoken':
        return tiktoken.get_encoding('gpt2')
    

def get_loss_fn(name: str):
    if name == 'CE':
        return F.cross_entropy


def get_profiler(profile: bool = False):
    if profile:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        schedule = torch.profiler.schedule(skip_first=0, wait=0, warmup=3, active=15, repeat=1)
        prof = torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
            with_flops=True,
            acc_events=True,
        )
        return prof
    else:
        return NullProfiler()


def get_logger(rank: int, args, train_config):
    if rank == 0:
        wandb.login(WANDB_API_KEY)
        run = wandb.init(
            name=args.run_name, 
            project='GPT2-Small', 
            dir=f'tmp/{args.run_name}',
            config=train_config,
        )
        return run
    else:
        return NullLogger()
    

def get_timestamp():
    return str(datetime.datetime.now()).replace(' ', '_').replace(':', '-').split('.')[0]

class NullProfiler():
    def __init__(self):
        pass

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc, tb):
        pass

    def step(self):
        pass


class NullLogger():
    def __init__(self):
        pass

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass