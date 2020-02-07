import torch
import h5py
from torch.utils.data.dataset import Dataset


class DiskDataset(Dataset):

    def __init__(self, args, domain_idx, examples_pth):
        self.domain_idx = domain_idx

        self.cache = []
        self.cache_len = args.get('cache_len', 100000)

        self.examples_hf = h5py.File('examples_pth', 'r')
        self.num_examples = self.examples_hf['examples'].shape[0]

    def __getitem__(self, index):
        _global = self.examples_hf['globals']['data'][index]
        _center = self.examples_hf['centers']['context_data'][index]
        _context = self.examples_hf['contexts']['context_data'][index]
        
        return (
            torch.tensor([self.domain_idx]),
            torch.tensor([_global]),
            torch.tensor([_center]),
            torch.tensor(_context)
        )

    def __len__(self):
        return self.num_examples
