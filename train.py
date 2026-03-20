import torch
import yaml
import wandb
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

    # logging & profiling
    wandb.login(WANDB_API_KEY)
    run = wandb.init(
        name=args.run_name, 
        project='GPT2-Small', 
        dir=f'tmp/{args.run_name}',
        config=train_config,
    )
    prof = get_profiler(args.profile)

    train_loss_list = []
    global_step = 0
    accum_loss = torch.tensor(0., device=device)
    for e in range(train_config['n_epochs']):
        pbar = tqdm(trainloader)
        with prof as prof:
            for step, (data, targets) in enumerate(pbar):
                prof.step()
                global_step += 1
                if args.profile and prof.schedule(step) == torch.profiler.ProfilerAction.NONE:
                    break

                loss = train_step(model, optim, global_step, data, targets, device, loss_fn, run)
                accum_loss += loss
                train_loss_list.append(loss)
                pbar.set_postfix({'epoch': e, 'loss': sum(train_loss_list[-10:]) / min(10, len(train_loss_list) + 1)})
                if step % args.log_every == 0:
                    train_loss_list.append(accum_loss.detach().item() / args.log_every)
                    accum_loss = 0

        if args.profile:
            prof.export_chrome_trace(f'tmp/train_trace_{get_timestamp()}.json')
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=30))
            break

        valid_loss = validate(model, loss_fn, validloader)
        run.log({'valid_loss': valid_loss}, step=global_step)
        print(f'For epoch {e} train loss was: {train_loss_list[-1]:.2f} and valid loss was {valid_loss:.2f}')
    
    run.finish()
    return model


def train_step(model, optim, step, data, targets, device, loss_fn, run):
    
    start_time = perf_counter()
    data, targets = data.to(device), targets.to(device)
    run.log({'data_load_time': perf_counter() - start_time}, step=step)

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
    run.log({'optim_step_time': perf_counter() - start_time}, step=step)
    optim.zero_grad()

    return loss


def validate(model, loss_fn, validloader: DataLoader):
    device = next(model.parameters()).device
    with torch.inference_mode():
        pbar = tqdm(validloader)
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = loss_fn(logits.flatten(0, 1), targets.flatten(0, 1), reduction='mean')
        return loss

        
def parse_args(args: list[str] = None):
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='GPT-2', help='Model type')
    parser.add_argument('--profile', action='store_true', help='Torch-profile training steps')
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--run_name')

    args = parser.parse_args(args)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)


