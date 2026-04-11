from __future__ import annotations
import tiktoken
import numpy as np
import torch
import copy
from torch.utils.data import Dataset


CORPUSES = {
    'shakespear_tiny': 'data/input.txt',
    'wikitext-103': 'data/wiki.train.raw.txt',
}

class TextDataset(Dataset):
    def __init__(self, tokenizer = None, corpus_name: str = None, max_seq_len: int = 64):
        if not corpus_name:
            return
        assert tokenizer and corpus_name, 'Must provide tokenizer and corpus name to initialize dataset'
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        data, target = self._get_input_and_target(CORPUSES[corpus_name])
        self._chunk_corpus_into_batches(data, target)
        

    def _chunk_corpus_into_batches(self, data, targets):
        """Chunk the tokenised data into training samples of max_seq_len"""
        n_samples = len(data) // self.max_seq_len
        self.data_samples = torch.tensor(
            [data[i * self.max_seq_len : (i + 1) * self.max_seq_len] for i in range(n_samples)], 
            dtype=torch.int64,
        )
        self.target_samples = torch.tensor(
            [targets[i * self.max_seq_len : (i + 1) * self.max_seq_len]for i in range(n_samples)], 
            dtype=torch.int64,
        )
        assert len(self.data_samples) == len(self.target_samples)


    def split_valid_from_train(self, fraction=0.1) -> TextDataset:
        """Return a random permutaion subset to form a validset; remove from trainset"""
        assert len(self.data_samples), 'Dataset is empty'
        perm = torch.randperm(len(self.data_samples))
        valid_frac = int(fraction * len(self.data_samples))

        valid_idxs = perm[:valid_frac].tolist()
        keep_mask = torch.fill(torch.empty(self.data_samples.shape[0], dtype=torch.bool), value=0)
        keep_mask[perm[valid_frac:]] = True    

        validset = TextDataset()
        validset.data_samples = copy.deepcopy(self.data_samples[valid_idxs])
        self.data_samples = self.data_samples[keep_mask]
        validset.target_samples = copy.deepcopy(self.target_samples[valid_idxs])
        self.target_samples = self.target_samples[keep_mask]

        return validset 


    def __len__(self):
        return len(self.data_samples)

    
    def __getitem__(self, index):
        return self.data_samples[index], self.target_samples[index]


    def _get_input_and_target(self, corpus_path: str) -> tuple[list[int], list[int]]:
        """
        Loads corpus into memory and tokenizes.
        Targets are inputs offset by one for next token prediction task.
        """
        eot = '<|endoftext|>'
        eot_id = self.tokenizer.encode(eot, allowed_special={'<|endoftext|>'})

        # start input with eot token to create offset by one 
        input = eot_id.copy()
        target = []
        with open(corpus_path) as r_h:
            for line in r_h.readlines():
                if line:
                    tokenised_line = self.tokenizer.encode(line)
                    input.extend(tokenised_line + eot_id)
                    target.extend(tokenised_line + eot_id)

        return input, target