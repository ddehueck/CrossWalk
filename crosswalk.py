import torch as t
import torch.nn as nn
import pandas as pd
import numpy as np


class PyPICrossWalk(nn.Module):

    def __init__(self, embed_len, domains):
        super().__init__()
        self.domains = nn.ModuleList(domains)
        self.embed_len = embed_len
        # self.id2name[0] --> pylineclip
        self.id2name = self.create_id2name('data/pypi_nodes.csv')
        # pylineclip -->  0
        self.name2id = {v: k for k, v in self.id2name.items()}
        self.entity_embeds = nn.Embedding(len(self.id2name), embed_len)

    def init_domains(self):
        for domain in self.domains:
            print(f'Initializing domain: {domain.name}...')
            domain.load_embeds()
            domain.set_local2global(self.id2name)
            if domain.is_local_global:
                domain.set_sgns(global_embeds=self.entity_embeds)
            else:
                domain.set_sgns()

    def get_local_embeds(self, domain_id, idxs):
        if self.domains[domain_id].is_local_global:  # idxs are already global ids
            return self.get_global_embeds(idxs)
        else:
            return self.domains[domain_id].embeds(idxs)

    def get_global_embeds(self, idxs):
        return self.entity_embeds(idxs)

    def calculate_local_loss(self, domain_id, global_embeds, center_embeds, context_embeds):
        # TODO: do averaging when also learning local embeddings
        if self.domains[domain_id].is_local_global:
            centers = center_embeds
        else:
            # Average - to combine - try concatenation, summation, seperate?
            centers = t.mean(t.stack((center_embeds, global_embeds)), dim=0)

        return self.domains[domain_id].sgns(centers, context_embeds)

    @staticmethod
    def create_id2name(nodes_path):
        nodes_names = pd.read_csv(nodes_path, na_filter=False)['nodes'].values
        return dict(zip(range(len(nodes_names)), nodes_names))

    def nearest_neighbors(self, node):
        """
        Finds vector closest to word_idx vector
        :param word: String
        :param dictionary: Gensim dictionary object
        :return: Integer corresponding to word vector in self.word_embeds
        """
        index = self.name2id[node]
        vectors = self.entity_embeds.weight.detach().cpu().numpy()
        query = vectors[index]

        ranks = vectors.dot(query).squeeze()
        denom = query.T.dot(query).squeeze()
        denom = denom * np.sum(vectors ** 2, 1)
        denom = np.sqrt(denom)
        ranks = ranks / denom
        mostSimilar = []
        [mostSimilar.append(idx) for idx in ranks.argsort()[::-1]]
        nearest_neighbors = mostSimilar[:6]
        nearest_neighbors = [self.id2name[comp] for comp in nearest_neighbors]

        return nearest_neighbors
