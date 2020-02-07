import torch
import h5py
from torch.utils.data.dataset import Dataset


class DiskDataset(Dataset):

    def __init__(self, args, examples_pth):
        self.cache = []
        self.cache_len = args.get('cache_len', 100000)

        self.examples_pth = examples_pth
        with h5py.File(examples_pth, 'r') as hf:
            self.num_examples = hf['globals']['data'].shape[0]
            print(f'This dataset has: {self.num_examples} examples.')

        self.init_cache(args.get('max_cache_prop', .5))

    def __getitem__(self, index):
        # Check if index is already in cache
        if index < self.max_cache_size:
            _global, _center, _context = self.cache[index]
        else:
            hf = h5py.File(self.examples_pth, 'r')
            _global = hf['globals']['data'][index]
            _center = hf['centers']['data'][index]
            _context = hf['contexts']['data'][index]

        return (
            torch.tensor([_global]),
            torch.tensor([_center]),
            torch.tensor(_context)
        )

    def init_cache(self, max_cache_prop):
        """
        One time operation so we don't have to do it every epoch!
        :param max_cache_prop: Float between 0.0 and 1.0
        :return: None - modifies self.cache
        """
        print(f'Creating cache of {max_cache_prop * 100}% of all examples in this domain...')
        assert 0.0 <= max_cache_prop <= 1.0
        self.max_cache_size = int(max_cache_prop * self.num_examples)

        with h5py.File(self.examples_pth, 'r') as hf:
            self.cache = list(zip(
                hf['globals']['data'][:self.max_cache_size],
                hf['centers']['data'][:self.max_cache_size],
                hf['contexts']['data'][:self.max_cache_size]
            ))

    def __len__(self):
        return self.num_examples
