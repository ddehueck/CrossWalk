from ..base_domain import BaseDomain
from ..disk_dataset import DiskDataset

from .example_utils import create_examples


class PyPIGraphDomain(BaseDomain):
    def __init__(self, args):
        super().__init__(args)

        self.name = 'PyPI Graph Domain'
        self.dataset = DiskDataset(args, 'examples.pth')
        self.examples, self.dictionary = create_examples(edgelist_path, n_walks, walk_len, window_size, save=True)

    def set_global2local(self, id2name):
        """ Node embeddings relate to global embeddings"""
        self.global2local = {}
        for node_id in self.G.nodes():
            self.global2local[int(node_id)] = self.dictionary.token2id[str(node_id)]

