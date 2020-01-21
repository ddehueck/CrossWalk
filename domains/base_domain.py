import jax
import torch
from .sgns_loss import SGNSLoss


class BaseDomain:
    """ Default values for all domains """

    def __init__(self, window_size, embed_len):
        self.window_size = window_size
        self.dictionary = None
        self.global2local = None  # Dictionary that relates local domain id to global entity id

        self.walks = []
        self.examples = []

        self.queries = []
        self.name = 'Unnamed Domain'

        self.embed_len = embed_len
        self.embeds = None
        self.sgns = None

    def set_global2local(self, global_dictionary):
        """
        Sets self.global2local to a dictionary
        Used to relate local domain embedding ids to global entity ids
        self.global2local[global_id] --> local_id
        """
        raise NotImplementedError

    def load_embeds(self):
        """ Needs to be called after domain has a dictionary """
        # This initialization is important!
        torch_embed = torch.nn.Embedding(len(self.dictionary), self.embed_len)
        self.embeds = jax.numpy.array(torch_embed.weight.detach().numpy())

    def set_sgns(self):
        self.sgns = SGNSLoss(dictionary=self.dictionary)
