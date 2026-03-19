import tiktoken
from torch.nn import functional as F

def get_tokenizer(name: str = 'tiktoken'):
    if name == 'tiktoken':
        return tiktoken.get_encoding('gpt2')
    
def get_loss_fn(name: str):
    if name == 'CE':
        return F.cross_entropy