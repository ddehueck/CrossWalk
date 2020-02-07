import torch.nn as nn
from .sgns_loss import SGNSLoss


class BaseDomain(nn.Module):
    """ Default values for all domains """

    def __init__(self, args):
        super().__init__()
        self.name = 'Unnamed Domain'
        self.dictionary = None
        self.dataset = None
        self.global2local = None  # Dictionary that relates local domain id to global entity id
        self.embed_len = args.get('embed_len')
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
        self.embeds = nn.Embedding(len(self.dictionary), self.embed_len)

    def set_sgns(self):
        self.sgns = SGNSLoss(dictionary=self.dictionary, embeds=self.embeds)
