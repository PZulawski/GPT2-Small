import os
import torch
import yaml
import math
import gc
from time import perf_counter
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model import Transformer
from dataset import TextDataset
from dataloader import PrefetchDataLoader
from util import get_tokenizer, get_loss_fn, get_profiler, get_logger, get_timestamp, setup, cleanup
from torch import nn
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP




def main(args):
    assert torch.cuda.is_available(), 'CUDA not found'
    world_size = torch.cuda.device_count()
    assert not (world_size == 1 and args.DDP), 'Attempting to run DDP on a single device'
    print(f'Running on {world_size} devices')

    # torch.compile with multi-thread and nsys don't interact well, limit to single thread compilation 
    if args.profile == 'nsys':
        os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

    mp.spawn(train, args=(world_size, args), nprocs=world_size, join=True)


def train(rank, world_size, args):
    setup(rank, world_size, args.DDP)
    device = torch.device(rank)
    # load model and training run parameters
    with open('model_defaults.yaml', 'r') as config:
        config = yaml.unsafe_load(config)
        train_config = config[args.model]['train_config']

    tokenizer = get_tokenizer(config[args.model]['tokenizer'])

    # fix torch seed to limit noise impact on reproducibility 
    if args.fixed_seed: 
        torch.manual_seed(42)
        gc.disable()
    
    # init model and loss
    model = Transformer(**config[args.model]['model_config'], vocab_size=tokenizer.n_vocab).to(device)
    model.compile(fullgraph=True, mode='reduce-overhead')
    if args.DDP:
        nn.SyncBatchNorm.convert_sync_batchnorm(model)
        distributed_model =  DDP(model, device_ids=[rank], bucket_cap_mb=100)
    else:
        distributed_model = model
    loss_fn = torch.compile(get_loss_fn(train_config['loss']), fullgraph=True, mode='reduce-overhead')

    # init data loading utilities
    assert train_config['batch_size'] % world_size == 0, 'Select effective batch size divisible by world size'
    batch_size_per_rank = train_config['batch_size'] // world_size
    trainset = TextDataset(tokenizer, corpus_name=args.corpus_name, max_seq_len=model.max_ctx)
    validset = trainset.split_valid_from_train(fraction=args.valid_fraction)
    if args.DDP:
        print(f'Per device batch size is {batch_size_per_rank}')
        sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        trainloader = DataLoader(trainset, batch_size_per_rank, sampler)
    else:
        print(f'Per device batch size is {batch_size_per_rank}')
        trainloader = DataLoader(trainset, batch_size_per_rank, shuffle=True)
    if args.prefetch_data and not args.DDP:
        trainloader = PrefetchDataLoader(trainloader, device)
    validloader = DataLoader(validset, batch_size=batch_size_per_rank, shuffle=False)

    # init optimisation utilities
    optim = torch.optim.AdamW(
        distributed_model.parameters(), 
        lr=train_config['lr'], 
        betas=(train_config['beta_linear'], train_config['beta_square']),
        weight_decay=train_config['weight_decay'],
        fused=True,
    )
    lr_scheduler = CosineDecayScheduler(
        optim, 
        lr_max=train_config['lr'],
        wu_fraction=train_config['wu_fraction'], 
        total_steps=len(trainloader) * train_config['n_epochs'] + 1,
    )    
    # init logging & profiling
    run = get_logger(rank, args, train_config)
    prof = get_profiler(args.profile)

    global_step = 0
    for e in range(train_config['n_epochs']):
        accum_loss = torch.tensor(0., device=device)
        pbar = tqdm(trainloader) if rank == 0 else trainloader
        with prof as prof:
            for step, (data, targets) in enumerate(pbar):
                prof.step()
                global_step += 1
                if args.profile == 'pytorch' and prof.schedule(step) == torch.profiler.ProfilerAction.NONE:
                    break

                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    loss = train_step(
                        args,
                        distributed_model, 
                        optim, lr_scheduler, 
                        global_step, 
                        data, 
                        targets, 
                        device, 
                        loss_fn, 
                        run,
                    )

                accum_loss += loss
                if rank == 0:
                    pbar.set_postfix({'epoch': e, 'loss': accum_loss / step})

                if global_step % 1000 == 0:
                    valid_loss = validate(rank, model, loss_fn, validloader)
                    run.log({'valid_loss': valid_loss}, step=global_step)
                    print(f'At step {global_step} valid loss was {valid_loss:.2f}')
                    save_model_checkpoint(rank, model, optim, global_step, valid_loss, args.run_name)
            

        if args.profile == 'pytorch':
            if rank == 0:
                prof.export_chrome_trace(f'tmp/train_trace_{get_timestamp()}.json')
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
                print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
            break
    
    save_model_checkpoint(rank, model, optim, global_step, valid_loss, args.run_name)
    cleanup(args.DDP)
    run.finish()
    return distributed_model    


def train_step(args, model, optim, lr_scheduler, step, data, targets, device, loss_fn, run):
    """Executes a single training step, records and logs telemetry to W&B"""
    
    start_time_step = perf_counter()
    data, targets = data.to(device), targets.to(device)
    run.log({'data_load_time': perf_counter() - start_time_step}, step=step)

    start_time = perf_counter()
    logits = model(data)
    run.log({'model_forward_time': perf_counter() - start_time}, step=step)

    start_time = perf_counter()
    loss = loss_fn(logits.flatten(0, 1), targets.flatten(0, 1), reduction='mean')
    run.log({'loss_time': perf_counter() - start_time}, step=step)
    run.log({'loss': loss}, step=step)

    start_time = perf_counter()
    loss.backward()
    run.log({'model_backward_time': perf_counter() - start_time}, step=step)

    start_time = perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optim.step()
    lr_scheduler.step()
    run.log(
        {
            'optim_step_time': perf_counter() - start_time, 
            'lr': lr_scheduler.get_last_lr()[0],
            'token_throughput': data.shape[0] * data.shape[1] / (perf_counter() - start_time_step),
        }, 
        step=step,
    )

    if step % args.log_every == 0 and args.log_acts_and_grads:
        log_acts_and_grads(run, model, step)
    optim.zero_grad()

    return loss


def save_model_checkpoint(rank, model, optim, global_step, valid_loss, run_name):
    if rank == 0:
        chkpt_dir = f'chkpts/{run_name}'
        os.makedirs(chkpt_dir, exist_ok=True)
        torch.save(
            {                                                                                                                                                                                                               
                'step': global_step,                                            
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),                                                                                                                                                                                
                'valid_loss': valid_loss,
            }, 
            f=f"{chkpt_dir}/checkpoint_step_{global_step}.pt",
        ) 
    

@torch.inference_mode()
def validate(rank, model, loss_fn, validloader: DataLoader):
    device = next(model.parameters()).device
    pbar = tqdm(validloader) if rank == 0 else validloader
    total_loss = 0
    for i, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        logits = model(data)
        total_loss += loss_fn(logits.flatten(0, 1), targets.flatten(0, 1), reduction='mean').item()
    return total_loss / (i + 1)
    

def log_acts_and_grads(run, model, step):
    run.log({f"weight_max/{name}": torch.max(params).item() for name, params in model.state_dict().items()}, step=step)
    run.log({f"grad_max/{name}": torch.max(params.grad).item() for name, params in model.named_parameters()}, step=step)

class CosineDecayScheduler(torch.optim.lr_scheduler.LRScheduler):
    """
    Cosine decay schduler, ramps up linearly from lr_min to lr_max over wu_fraction of total_steps, then 
    follows a cosine curve from lr_max back down to lr_min
    """

    def __init__(self, optim, lr_max, wu_fraction, total_steps, lr_min = 0):
        self.current_step = 0
        self.lr_max = lr_max
        self.wu_fraction = wu_fraction
        self.total_steps = total_steps
        self.wu_steps = self.total_steps * self.wu_fraction
        self.lr_min = (lr_max / 100) if lr_min == 0 else lr_min
        super().__init__(optim)

    def step(self):
        self.current_step += 1
        assert self.current_step <= self.total_steps
        super().step()

    def get_lr(self):
        adjusted_scale = (self.lr_max - self.lr_min)
        if self.current_step <= self.wu_steps:
            return [(self.current_step / self.wu_steps) * adjusted_scale + self.lr_min]
        else:
            adjusted_cosine_step = (self.current_step - self.wu_steps) / (self.total_steps - self.wu_steps)
            return [math.cos((adjusted_cosine_step) * math.pi / 2) * adjusted_scale + self.lr_min]

      
def parse_args(args: list[str] = None):
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='GPT-2', help='Model type, picks params from model_defaults.yaml')
    parser.add_argument(
        '--corpus_name', 
        type=str, 
        choices=['shakespear_tiny', 'wikitext-103'],
        default='shakespear_tiny', 
        help='Name of training corpus',
    )
    parser.add_argument('--valid_fraction', type=float, default=0.001, help='Fraction of data set used for validation')
    parser.add_argument('--prefetch_data', action='store_true')
    parser.add_argument(
        '--profile', 
        type=str, 
        choices=['nsys', 'pytorch', 'none'], 
        default='none', 
        help='Type of profiler to be used; nsys, pytorch or none (default)'
    )
    parser.add_argument('--log_every', type=int, default=10, help='Number of steps between logging')
    parser.add_argument('--run_name', type=str, help='Name of the run for W&B logging')
    parser.add_argument('--fixed_seed', action='store_true', help='Set to reduce variance in profiling reproducibility')
    parser.add_argument('--log_acts_and_grads', action='store_true', help='Set to log activation and grad max to W&B')
    parser.add_argument('--DDP', action='store_true', help='Set to run Distribute Data Parallel')

    args = parser.parse_args(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)


