import tiktoken
import torch
import datetime
from torch.nn import functional as F

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