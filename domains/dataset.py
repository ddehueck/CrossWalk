import numpy as np
from torch.utils.data.dataset import Dataset


class CrossWalkDataset(Dataset):

    def __init__(self, domains):
        self.domains = domains
        self.domain_lens = [len(d.examples) for d in domains]

    def __getitem__(self, index):
        for d_idx, d_len in enumerate(self.domain_lens):
            if d_len < index:
                index -= d_len

        example = self.domains[d_idx].examples[index]
        return np.array(d_idx), np.array(example)

    def __len__(self):
        return sum(self.domain_lens)
