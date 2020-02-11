import torch
import pandas as pd
from gensim.corpora import Dictionary

from ..base_domain import BaseDomain
from ..disk_dataset import DiskDataset

from .utils import load_examples


class PyPILanguageDomain(BaseDomain):

    def __init__(self, args, df_path):
        BaseDomain.__init__(self, args)
        self.name = 'PyPI Language Domain'

        examples_pth, dictionary_pth = load_examples(args, df_path)
        self.dataset = DiskDataset(args, examples_pth)
        self.dictionary = Dictionary().load(dictionary_pth)

    def set_local2global(self, id2name):
        """ Document embeddings relate to global """
        self.local2global = {}
        for id in id2name.keys():
            self.local2global[id] = id