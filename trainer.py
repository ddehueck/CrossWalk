import torch as t
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from functools import partial
import numpy as np
from tqdm import tqdm

from crosswalk import PyPICrossWalk
from domains.dataset import CrossWalkDataset
from domains.graph_domain import PyPIGraphDomain
from domains.language_domain import PyPILanguageDomain


class PyPITrainer:

    def __init__(self):
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.crosswalk = PyPICrossWalk(
            embed_len=128,
            domains=[
                PyPIGraphDomain(5, 128, 'data/pypi_edges.csv', 10, 40),
                PyPILanguageDomain(5, 128, 'data/pypi_lang.csv')
            ]
        )
        self.crosswalk.init_domains()
        self.dataset = CrossWalkDataset(self.crosswalk.domains)
        self.dataloader = DataLoader(self.dataset, batch_size=2048, shuffle=True, num_workers=2)
        self.optimizer = optim.Adam(self.crosswalk.parameters(), lr=1e-3)

    def train(self):
        print(f'Training on: {self.device}')
        self.crosswalk.to(self.device)
        
        for epoch in range(100):
            print(f'Beginning epoch: {epoch + 1}/100')
            for i, batch in enumerate(tqdm(self.dataloader)):
                # Unpack Data
                domain_ids, global_ids, center_ids, contexts_ids = batch
                # Send to device
                global_ids = global_ids.to(self.device)
                center_ids = center_ids.to(self.device)
                contexts_ids = contexts_ids.to(self.device)
                 # Remove accumulated gradients
                self.optimizer.zero_grad()
                # Split batch up by domain and update domain's weights
                batch_idxs_by_domain = [t.where(domain_ids == i)[0] for i in range(len(self.crosswalk.domains))]
                for d_idx, batch_idxs in enumerate(batch_idxs_by_domain):
                    if len(batch_idxs) == 0: continue
                    # Get domain's embeddings
                    context_embeds = self.crosswalk.get_local_embeds(d_idx, contexts_ids[batch_idxs])
                    center_embeds = self.crosswalk.get_local_embeds(d_idx, center_ids[batch_idxs])
                    # Get global embeddings
                    global_embeds = self.crosswalk.get_global_embeds(global_ids[batch_idxs])
                    # Calculate loss
                    loss = self.crosswalk.calculate_local_loss(d_idx, global_embeds, center_embeds, context_embeds)
                    # Backprop but don't step!
                    loss.backward()
                # Update - outside of loss loop  so gradients don't influence eachother in one batch!
                self.optimizer.step()
                
            self.log_step(epoch)

    def log_step(self, epoch):
        print(f'EPOCH: {epoch} | GRAD: {t.sum(self.crosswalk.entity_embeds.weight.grad)}')
        # Log embeddings!
        print('\nLearned embeddings:')
        for n in ['torch', 'tensorflow', 'flask', 'django', 'numpy']:
            print(f'Node: {n} neighbors: {self.crosswalk.nearest_neighbors(n)}')
        print()

        for q in self.crosswalk.domains[1].queries:
            print(f'Word: {q} neighbors: {self.crosswalk.domains[1].nearest_neighbors(q)}')
        print()


if __name__ == '__main__':
    trainer = PyPITrainer()
    trainer.train()
