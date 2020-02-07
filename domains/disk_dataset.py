import torch
import h5py
from torch.utils.data.dataset import Dataset


class CrossWalkDataset(Dataset):

    def __init__(self, domains):
        self.domains = domains
        self.domain_lens = [len(d.examples) for d in domains]
        
        self.cache = []
        self.cache_len = 100000

        self.examples_hf = h5py.File('all_examples.h5', 'r')


    def __getitem__(self, index):
        for d_idx, d_len in enumerate(self.domain_lens):
            if d_len < index:
                index -= d_len

        _global, _center, _context = self.domains[d_idx].examples[index]
        return (
            torch.tensor([d_idx]), 
            torch.tensor([_global]), 
            torch.tensor([_center]), 
            torch.tensor(_context)
        )

    def __len__(self):
        return sum(self.domain_lens)
