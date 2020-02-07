from gensim.corpora import Dictionary

from ..base_domain import BaseDomain
from ..disk_dataset import DiskDataset

from .example_utils import create_examples


class PyPIGraphDomain(BaseDomain):
    def __init__(self, args, edgelist_path):
        super().__init__(args)
        self.name = 'PyPI Graph Domain'

        examples_pth, dictionary_pth = create_examples(args, edgelist_path, save=True)
        self.dataset = DiskDataset(args, examples_pth)
        self.dictionary = Dictionary().load(dictionary_pth)

    def set_global2local(self, id2name):
        """ Node embeddings relate to global embeddings"""
        self.global2local = {}
        for node_id in self.G.nodes():
            self.global2local[int(node_id)] = self.dictionary.token2id[str(node_id)]
