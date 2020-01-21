import jax.numpy as np

from domains.utils import AliasMultinomial


class SGNSLoss:
    BETA = 0.75  # exponent to adjust sampling frequency
    NUM_SAMPLES = 2

    def __init__(self, dictionary):
        self.vocab_len = len(dictionary)
        # Helpful values for unigram distribution generation
        # Should use cfs instead but: https://github.com/RaRe-Technologies/gensim/issues/2574
        self.transformed_freq_vec = np.array([dictionary.dfs[i] for i in range(self.vocab_len)]) ** self.BETA
        self.freq_sum = np.sum(self.transformed_freq_vec)
        # Generate table
        self.unigram_table = self.generate_unigram_table()

    def forward(self, global_params, local_params, contexts):
        # Unpack data
        print(contexts)
        print(contexts[:, 0])
        print(contexts[:, 2:])
        global_ids, center_ids, context_ids = contexts[:, 0], contexts[:, 1], contexts[:, 2:]
        print(global_ids)
        print(context_ids)
        # Get actual embeddings
        global_embeds = global_params[global_ids]
        center_embeds, context_embeds = local_params[center_ids], local_params[context_ids]
        # Squeeze into dimensions we want
        # center - [batch_size x embed_size]
        # context - [batch_size x window_size x embed_size]
        global_embeds = global_embeds.squeeze()
        center, context = center_embeds.squeeze(), context_embeds.squeeze()  # batch_size x embed_size
        # Mean all vectors used to predict contexts
        center = np.mean((center, global_embeds), axis=0)

        # Compute true portion
        true_scores = (np.expand_dims(center, axis=1) * context).sum(-1)  # batch_size
        loss = self.bce_loss_w_logits(true_scores, np.ones_like(true_scores))

        # Compute sampled portion
        samples = self.get_unigram_samples(bs=len(true_scores), ws=5, embeds=local_params)
        sample_dots = (np.expand_dims(np.expand_dims(center, axis=1), axis=1) * samples).sum(-1)
        loss += self.bce_loss_w_logits(sample_dots, np.zeros_like(sample_dots))

        return loss

    @staticmethod
    def bce_loss_w_logits(x, y):
        max_val = np.clip(x, 0, None)
        loss = x - x * y + max_val + np.log(np.exp(-max_val) + np.exp((-x - max_val)))
        return loss.mean()

    def get_unigram_samples(self, bs, ws, embeds):
        """
        Returns a sample according to a unigram distribution
        Randomly choose a value from self.unigram_table
        """
        # How many samples are needed
        window_size = ws * 2
        n = bs * self.NUM_SAMPLES * window_size
        embed_len = len(word_embeds[0])

        rand_idxs = self.unigram_table.draw(n)
        rand_idxs = rand_idxs.reshape(bs, window_size, self.NUM_SAMPLES)
        return word_embeds[rand_idxs].squeeze().reshape(bs, window_size, self.NUM_SAMPLES, embed_len)

    def get_unigram_prob(self, token_idx):
        return (self.transformed_freq_vec[token_idx].item()) / self.freq_sum.item()

    def generate_unigram_table(self):
        # Probability at each index corresponds to probability of selecting that token
        pdf = [self.get_unigram_prob(t_idx) for t_idx in range(0, self.vocab_len)]
        # Generate the table from PDF
        return AliasMultinomial(pdf)
