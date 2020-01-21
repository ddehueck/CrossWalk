from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.utils import strided_windows

from .base_domain import BaseDomain
from .graph_utils import load_pd_edgelist, build_random_walk_corpus


class PyPIGraphDomain(BaseDomain):
    def __init__(self, window_size, embed_len, edgelist_path, n_walks, walk_len):
        super().__init__(window_size=window_size, embed_len=embed_len)

        self.name = 'PyPI Graph Domain'
        self.G = load_pd_edgelist(edgelist_path)
        self.walks = self.generate_walks(n_walks, walk_len)

    def load_examples(self):
        for walk in tqdm(enumerate(self.walks), desc='Generating Examples:', total=len(self.walks)):
            windows = strided_windows(walk, self.window_size)
            for w in windows:
                center, context = w[0], w[1:]  # Add entity id as well
                self.examples.append([walk[0], center, context])

    def set_global2local(self, id2name):
        """ Node embeddings relate to global embeddings"""
        self.global2local = {}
        for node_id in self.G.nodes():
            self.global2local[int(node_id)] = self.dictionary.token2id[str(node_id)]

    def generate_walks(self, n_walks, walk_len):
        walks = build_random_walk_corpus(self.G, n_walks, walk_len)
        # Now we have a Gensim Dictionary to work with
        self.dictionary = Dictionary(walks)
        # Covert docs to indexes in dictionary
        return [self.dictionary.doc2idx(w) for w in tqdm(walks, desc='Converting to indicies')]

