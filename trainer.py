from functools import partial

import jax.experimental.optimizers as optim
import numpy as np
from jax import jit, grad
from tqdm import tqdm

from crosswalk import PyPICrossWalk
from dataloader import NumpyLoader
from domains.dataset import CrossWalkDataset
from domains.graph_domain import PyPIGraphDomain
from domains.language_domain import PyPILanguageDomain


class PyPITrainer:

    def __init__(self):
        self.crosswalk = PyPICrossWalk(
            embed_len=32,
            domains=[
                PyPIGraphDomain(5, 32, 'data/pypi_edges.csv', 1, 40),
                PyPILanguageDomain(5, 32, 'data/pypi_lang.csv')
            ]
        )
        self.crosswalk.init_domains()
        self.dataset = CrossWalkDataset(self.crosswalk.domains)
        self.dataloader = NumpyLoader(self.dataset, batch_size=1048, shuffle=True, num_workers=4)
        # Set up optimizer - rmsprop seems to work the best
        self.opt_init, self.opt_update, self.get_params = optim.adam(1e-3)

    #@partial(jit, static_argnums=(0,))
    def update(self, i, opt_state, domain_id, contexts):
        params = self.get_params(opt_state)
        loss_fn = self.crosswalk.domains[domain_id].sgns.forward
        g = grad(loss_fn)(params[0], params[domain_id], contexts)
        return self.opt_update(i, g, opt_state), params, g

    def train(self):
        # Initialize optimizer state!
        all_params = [self.crosswalk.entity_embeds] + [d.embeds for d in self.crosswalk.domains]
        opt_state = self.opt_init(all_params)

        for epoch in range(100):
            print(f'Beginning epoch: {epoch + 1}/100')
            for i, batch in enumerate(tqdm(self.dataloader)):
                domain_ids, contexts = batch

                batch_idxs_by_domain = [np.where(domain_ids == i)[0] for i in range(len(self.crosswalk.domains))]
                for d_idx, batch_idxs in enumerate(batch_idxs_by_domain):
                    if len(batch_idxs) == 0: continue
                    
                    domain_contexts = contexts[batch_idxs]
                    opt_state, params, g = self.update(i + d_idx, opt_state, d_idx, domain_contexts)

            self.log_step(epoch, params, g)

    def log_step(self, epoch, params, g):
        print(f'EPOCH: {epoch} | GRAD MAGNITUDE: {np.sum(g)}')
        # Log embeddings!
        print('\nLearned embeddings:')
        for word in self.dataset.queries:
            print(f'word: {word} neighbors: {self.crosswalk.nearest_neighbors(word, self.dataset.dictionary, params)}')


if __name__ == '__main__':
    trainer = PyPITrainer()
    trainer.train()
