import torch
from torch.utils.data.dataset import Dataset


class CrossWalkDataset(Dataset):

    def __init__(self, domains):
        self.domains = domains
        self.domain_lens = [len(d.dataset) for d in domains]

    def __getitem__(self, index):
        for d_idx, d_len in enumerate(self.domain_lens):
            if d_len <= index:
                index -= d_len
            else:
                break

        _global, _center, _context = self.domains[d_idx].dataset[index]
        # Convert global (which is stored as local ids) to truly global ids
        #_global = self.domains[d_idx].local2global[_global] # TODO: Is this even needed?

        return d_idx, _global, _center, _context

    def __len__(self):
        return sum(self.domain_lens)
