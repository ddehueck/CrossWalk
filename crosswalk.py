import jax
import torch
import pandas as pd
import numpy as onp


class PyPICrossWalk:

    def __init__(self, embed_len, domains):
        self.domains = domains
        self.embed_len = embed_len

        # self.id2name[0] --> pylineclip
        self.id2name = self.create_id2name('data/pypi_nodes.csv')

        # This initialization is important!
        torch_embed = torch.nn.Embedding(len(self.id2name), embed_len)
        self.entity_embeds = jax.numpy.array(torch_embed.weight.detach().numpy())

    def init_domains(self):
        print('Generating examples...')
        for domain in self.domains:
            domain.load_examples()

        print('Creating domain-specific embeddings...')
        for domain in self.domains:
            domain.load_embeds()

        print('Creating skigram losses...')
        for domain in self.domains:
            domain.set_sgns()

        print('Generating globa2local dicts...')
        for domain in self.domains:
            domain.set_global2local(self.id2name)

    @staticmethod
    def create_id2name(nodes_path):
        nodes_names = pd.read_csv(nodes_path, na_filter=False)['nodes'].values
        return dict(zip(range(len(nodes_names)), nodes_names))

    @staticmethod
    def nearest_neighbors(entity, dictionary, vectors):
        """
        Finds vector closest to word_idx vector
        :param word: String
        :param dictionary: Gensim dictionary object
        :return: Integer corresponding to word vector in self.word_embeds
        """
        index = dictionary.token2id[entity]
        query = vectors[index]

        ranks = vectors.dot(query).squeeze()
        denom = query.T.dot(query).squeeze()
        denom = denom * onp.sum(vectors ** 2, 1)
        denom = onp.sqrt(denom)
        ranks = ranks / denom
        mostSimilar = []
        [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
        nearest_neighbors = mostSimilar[:10]
        nearest_neighbors = [dictionary[comp] for comp in nearest_neighbors]

        return nearest_neighbors

