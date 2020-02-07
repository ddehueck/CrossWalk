from gensim.corpora import Dictionary

from ..base_domain import BaseDomain
from ..disk_dataset import DiskDataset

from .utils import load_examples
from .graph import load_pd_edgelist


class PyPIGraphDomain(BaseDomain):
    def __init__(self, args, edgelist_path):
        super().__init__(args)
        self.name = 'PyPI Graph Domain'
        self.G = load_pd_edgelist(edgelist_path)

        examples_pth, dictionary_pth = load_examples(args, edgelist_path, self.G)
        self.dataset = DiskDataset(args, examples_pth)
        self.dictionary = Dictionary().load(dictionary_pth)

    def set_global2local(self, id2name):
        """ Node embeddings relate to global embeddings"""
        self.global2local = {}
        for node_id in self.G.nodes():
            self.global2local[int(node_id)] = self.dictionary.token2id[str(node_id)]
