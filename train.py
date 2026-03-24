import torch
import yaml
import wandb
import math
from config_local import WANDB_API_KEY
from time import perf_counter
from tqdm import tqdm
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from model import Transformer
from dataset import TextDataset
from util import get_tokenizer, get_loss_fn, get_profiler, get_timestamp

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Running on device: {device}")

    with open('model_defaults.yaml', 'r') as config:
        config = yaml.unsafe_load(config)
        train_config = config[args.model]['train_config']

    tokenizer = get_tokenizer(config[args.model]['tokenizer'])
    if args.fixed_seed:
        torch.manual_seed(42)
    model = Transformer(**config[args.model]['model_config'], vocab_size=tokenizer.n_vocab).to(device)
    loss_fn = get_loss_fn(train_config['loss'])

    trainset = TextDataset(tokenizer, corpus_name='shakespear_tiny', max_seq_len=model.max_ctx)
    validset = trainset.split_valid_from_train()
    trainloader = DataLoader(trainset, batch_size=train_config['batch_size'], shuffle=True)
    validloader = DataLoader(validset, batch_size=train_config['batch_size'] * 3, shuffle=False)

    optim = torch.optim.Adam(
        model.parameters(), 
        lr=train_config['lr'], 
        betas=(train_config['beta_linear'], train_config['beta_square']),
        weight_decay=train_config['weight_decay'],
    )

    lr_scheduler = CosineDecayScheduler(
        optim, 
        lr_max=train_config['lr'],
        wu_fraction=train_config['wu_fraction'], 
        total_steps=len(trainloader) * train_config['n_epochs'] + 1,
    )

    # logging & profiling
    wandb.login(WANDB_API_KEY)
    run = wandb.init(
        name=args.run_name, 
        project='GPT2-Small', 
        dir=f'tmp/{args.run_name}',
        config=train_config,
    )
    prof = get_profiler(args.profile)

    global_step = 0
    for e in range(train_config['n_epochs']):
        accum_loss = torch.tensor(0., device=device)
        pbar = tqdm(trainloader)
        with prof as prof:
            for step, (data, targets) in enumerate(pbar):
                prof.step()
                global_step += 1
                if args.profile and prof.schedule(step) == torch.profiler.ProfilerAction.NONE:
                    break

                loss = train_step(model, optim, lr_scheduler, global_step, data, targets, device, loss_fn, run)
                accum_loss += loss
                pbar.set_postfix({'epoch': e, 'loss': accum_loss / step})

        if args.profile:
            prof.export_chrome_trace(f'tmp/train_trace_{get_timestamp()}.json')
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
            break

        valid_loss = validate(model, loss_fn, validloader)
        run.log({'valid_loss': valid_loss}, step=global_step)
        print(f'For epoch {e} valid loss was {valid_loss:.2f}')
    
    run.finish()
    return model


def train_step(model, optim, lr_scheduler, step, data, targets, device, loss_fn, run):
    
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
    

@torch.inference_mode()
def validate(model, loss_fn, validloader: DataLoader):
    device = next(model.parameters()).device
    with torch.inference_mode():
        pbar = tqdm(validloader)
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = loss_fn(logits.flatten(0, 1), targets.flatten(0, 1), reduction='mean')
        return loss
    

def log_acts_and_grads(run, model, step):
    run.log({f"weight_max/{name}": torch.max(params).item() for name, params in model.state_dict().items()}, step=step)
    run.log({f"grad_max/{name}": torch.max(params.grad).item() for name, params in model.named_parameters()}, step=step)

class CosineDecayScheduler(torch.optim.lr_scheduler.LRScheduler):
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
    parser.add_argument('--model', type=str, default='GPT-2', help='Model type')
    parser.add_argument('--profile', action='store_true', help='Torch-profile training steps')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--run_name')
    parser.add_argument('--fixed_seed', action='store_true')
    parser.add_argument('--log_acts_and_grads', action='store_true')

    args = parser.parse_args(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)


