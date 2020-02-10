import torch.nn as nn
from .sgns_loss import SGNSLoss


class BaseDomain(nn.Module):
    """ Default values for all domains """

    def __init__(self, args):
        super().__init__()
        self.name = 'Unnamed Domain'
        self.dictionary = None
        self.dataset = None
        self.embed_len = args.get('embed_len')
        self.embeds = None
        self.sgns = None

    def load_embeds(self):
        """ Needs to be called after domain has a dictionary """
        self.embeds = nn.Embedding(len(self.dictionary), self.embed_len)
        #self.embeds.weight.requires_grad = False  # TODO: Make this optional

    def set_sgns(self):
        self.sgns = SGNSLoss(dictionary=self.dictionary, embeds=self.embeds)
