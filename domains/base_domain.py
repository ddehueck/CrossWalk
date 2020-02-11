import torch.nn as nn
from .sgns_loss import SGNSLoss


class BaseDomain(nn.Module):
    """ Default values for all domains """

    def __init__(self, args):
        super().__init__()
        self.name = 'Unnamed Domain'
        self.is_local_global = False
        self.dictionary = None
        self.dataset = None
        self.embed_len = args.get('embed_len')
        self.embeds = None
        self.sgns = None

    def load_embeds(self):
        """ Needs to be called after domain has a dictionary """
        if self.is_local_global:
            return
        self.embeds = nn.Embedding(len(self.dictionary), self.embed_len)
        #self.embeds.weight.requires_grad = False  # TODO: Make this optional

    def set_sgns(self, global_embeds=None):
        if global_embeds is not None:
            # When is local global is true
            self.sgns = SGNSLoss(dictionary=self.dictionary, embeds=global_embeds)
        else:
            self.sgns = SGNSLoss(dictionary=self.dictionary, embeds=self.embeds)

    def set_local2global(self, global_dictionary):
        """
        Sets self.set_local2global to a dictionary
        Used to relate local domain embedding ids to global entity ids
        self.set_local2global[local_id] --> global_id
        """
        raise NotImplementedError
