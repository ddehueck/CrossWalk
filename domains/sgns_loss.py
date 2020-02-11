import torch
import numpy as np
import torch.nn as nn

from domains.utils import AliasMultinomial


class SGNSLoss(nn.Module):
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 2

    def __init__(self, dictionary, embeds):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.vocab_len = len(dictionary)
        self.embeds = embeds  # Should this be here? - does it get updated through training?
        # Helpful values for unigram distribution generation
        # Should use cfs instead but: https://github.com/RaRe-Technologies/gensim/issues/2574
        self.transformed_freq_vec = torch.tensor(
            np.array([dictionary.dfs[i] for i in range(self.vocab_len)]) ** self.BETA
        )
        self.freq_sum = torch.sum(self.transformed_freq_vec)
        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, center, context):
        # Squeeze into dimensions we want
        # center - [batch_size x embed_size]
        # context - [batch_size x window_size x embed_size]
        center, context = center.squeeze(), context.squeeze()  # batch_size x embed_size
        
        # Compute true portion
        true_scores = (center.unsqueeze(1) * context).sum(-1)  # batch_size
        loss = self.criterion(true_scores, torch.ones_like(true_scores))

        # Compute sampled portion
        samples = self.get_unigram_samples(bs=len(true_scores), ws=5)
        sample_dots = (center.unsqueeze(1).unsqueeze(1) * samples).sum(-1)
        loss += self.criterion(sample_dots, torch.zeros_like(sample_dots))

        return loss

    def get_unigram_samples(self, bs, ws):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        # How many samples are needed
        window_size = ws * 2
        n = bs * self.NUM_SAMPLES * window_size
        embed_len = self.embeds.weight.data.shape[1]

        rand_idxs = self.unigram_table.draw(n)
        rand_idxs = rand_idxs.reshape(bs, window_size, self.NUM_SAMPLES)
        return self.embeds(rand_idxs).squeeze().view(bs, window_size, self.NUM_SAMPLES, embed_len)

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        # Probability at each index corresponds to probability of selecting that token
        pdf = [self.get_unigram_prob(t_idx) for t_idx in range(0, self.vocab_len)]
        # Generate the table from PDF
        return AliasMultinomial(pdf)
