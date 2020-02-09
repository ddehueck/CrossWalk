import torch as t
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from crosswalk import PyPICrossWalk
from domains.dataset import CrossWalkDataset
from domains.graph.domain import PyPIGraphDomain
from domains.language.domain import PyPILanguageDomain


class PyPITrainer:

    def __init__(self, args):
        self.device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
        self.writer = SummaryWriter(log_dir='./experiments/', flush_secs=3)

        self.crosswalk = PyPICrossWalk(
            embed_len=args['embed_len'],
            domains=[
                #PyPIGraphDomain(args, 'data/pypi_edges.csv'),
                PyPILanguageDomain(args, 'data/pypi_lang.csv')
            ]
        )

        self.crosswalk.init_domains()
        self.dataset = CrossWalkDataset(self.crosswalk.domains)
        self.dataloader = DataLoader(self.dataset, batch_size=8192, shuffle=True, num_workers=2)
        self.optimizer = optim.Adam(self.crosswalk.parameters(), lr=1e-3)

        #print('Just going over the data...')
        #for b in tqdm(self.dataloader):
        #    continue

    def train(self):
        print(f'Training on: {self.device}')
        self.crosswalk.to(self.device)

        for epoch in range(50):
            print(f'Beginning epoch: {epoch + 1}/50')
            running_loss = 0.0
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
                running_loss += loss.item()

            self.log_step(epoch + 1)
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, running_loss/len(self.dataloader))

    def log_step(self, epoch):
        print(f'EPOCH: {epoch} | GRAD: {t.sum(self.crosswalk.entity_embeds.weight.grad)}')
        # Log embeddings!
        print('\nLearned embeddings:')
        for n in ['torch', 'tensorflow', 'flask', 'django', 'numpy']:
            print(f'Node: {n} neighbors: {self.crosswalk.nearest_neighbors(n)}')
        print()

    def save_checkpoint(self, epoch, loss):
        # To visualize embeddings
        self.writer.add_embedding(
            self.crosswalk.entity_embeds.weight,
            global_step=epoch,
            tag=f'epoch_{epoch}',
        )

        # Save checkpoint
        print(f'Beginning to save checkpoint')
        t.save({
            'epoch': epoch,
            'model_state_dict': self.crosswalk.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, f'checkpoints/epoch_{epoch}_ckpt.pth')
        print(f'Finished saving checkpoint')


if __name__ == '__main__':
    args = {
        'walk_len': 40,
        'n_walks': 80,
        'window_size': 5,
        'embed_len': 128
    }

    trainer = PyPITrainer(args)
    trainer.train()
