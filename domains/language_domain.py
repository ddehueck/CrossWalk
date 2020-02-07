import re
import torch
from tqdm import tqdm
import pandas as pd
from gensim.utils import strided_windows
from gensim.corpora import Dictionary

from .base_domain import BaseDomain


class PyPILanguageDomain(BaseDomain):

    def __init__(self, window_size, embed_len, df_path):
        BaseDomain.__init__(self, window_size=window_size, embed_len=embed_len)
        self.name = 'PyPI Language Domain'
        self.queries = ['tensorflow', 'pytorch', 'nlp', 'performance', 'encryption']
        self.walks = self.tokenize_files(df_path)

    def load_examples(self):
        for i, walk in tqdm(enumerate(self.walks), desc='Generating Examples:', total=len(self.walks)):
            windows = strided_windows(walk, self.window_size)
            for w in windows:
                center, context = w[0], w[1:]  # Add entity id as well
                self.examples.append([i, center, context])

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

    def tokenize_files(self, df_path):
        data = pd.read_csv(df_path, na_filter=False)['language'].values
        cleaned_docs = [self.tokenize_doc(f) for f in tqdm(data, desc='Tokenizing Docs')]
        # Now we have a Gensim Dictionary to work with
        self.dictionary = Dictionary(cleaned_docs)
        # Remove any tokens with a frequency less than 10
        self.dictionary.filter_extremes(no_below=10, no_above=0.75)
        # Covert docs to indexes
        indexed_docs = [self.dictionary.doc2idx(d) for d in tqdm(cleaned_docs, desc='Converting to indicies')]
        # Remove out of vocab tokens
        return [[t for t in d if t != -1] for d in tqdm(indexed_docs, desc="Removing out-of-vocab tokens")]

    @staticmethod
    def tokenize_doc(doc):
        # Convert to text make lowercase
        clean_doc = [t.lower().strip() for t in doc.split()]
        # Only allow characters in the alphabet, '_', and digits
        clean_doc = [re.sub('[^a-zA-Z0-9]', '', t) for t in clean_doc]
        # Remove any resulting empty indices
        return [t for t in clean_doc if len(t) > 0]

    def nearest_neighbors(self, word):
        """
        Finds vector closest to word_idx vector
        :param word_idx: Integer
        :return: Integer corresponding to word vector in self.word_embeds
        """
        vectors = self.embeds.weight.data.detach().cpu().numpy()
        index = self.dictionary.token2id[word]
        query = vectors[index]

        ranks = vectors.dot(query).squeeze()
        denom = query.T.dot(query).squeeze()
        denom = denom * np.sum(vectors ** 2, 1)
        denom = np.sqrt(denom)
        ranks = ranks / denom
        mostSimilar = []
        [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
        nearest_neighbors = mostSimilar[:8]
        nearest_neighbors = [self.dictionary[comp] for comp in nearest_neighbors]

        return nearest_neighbors
