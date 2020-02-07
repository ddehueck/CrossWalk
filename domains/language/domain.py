import torch
from gensim.corpora import Dictionary

from ..base_domain import BaseDomain
from ..disk_dataset import DiskDataset

from .utils import create_examples


class PyPILanguageDomain(BaseDomain):

    def __init__(self, args, df_path):
        BaseDomain.__init__(self, args)
        self.name = 'PyPI Language Domain'
        self.queries = ['tensorflow', 'pytorch', 'nlp', 'performance', 'encryption']

        examples_pth, dictionary_pth = create_examples(args, df_path, save=True)
        self.dataset = DiskDataset(args, examples_pth)
        self.dictionary = Dictionary().load(dictionary_pth)

    def set_global2local(self, id2name):
        """ Document embeddings relate to global """
        self.global2local = {}
        for id in id2name.keys():
            self.global2local[id] = id + len(self.dictionary)  # Offset by word embeds

    def load_embeds(self):
        """ Needs to be called after domain has a dictionary """
        # This initialization is important!
        # Word + Document embeddings! - Doc embeds are appended to word embeds
        self.embeds = torch.nn.Embedding(len(self.dictionary) + len(self.walks), self.embed_len)
