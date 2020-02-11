from io import open
from tqdm import tqdm
from six.moves import range
from six import iterkeys
from collections import defaultdict, Iterable
import random


class Graph(defaultdict):

    def __init__(self):
        super(Graph, self).__init__(list)

    def nodes(self):
        return self.keys()

    def adjacency_iter(self):
        return self.iteritems()

    def subgraph(self, nodes={}):
        subgraph = Graph()
        for n in nodes:
            if n in self:
                subgraph[n] = [x for x in self[n] if x in nodes]

        return subgraph

    def make_undirected(self):
        for v in self.keys():
            for other in self[v]:
                if v != other:
                    self[other].append(v)

        self.make_consistent()
        return self

    def make_consistent(self):
        for k in iterkeys(self):
            self[k] = list(sorted(set(self[k])))
        self.remove_self_loops()

        return self

    def remove_self_loops(self):
        removed = 0
        for x in self:
            if x in self[x]:
                self[x].remove(x)
                removed += 1
        return self

    def check_self_loops(self):
        for x in self:
            for y in self[x]:
                if x == y:
                    return True
        return False

    def has_edge(self, v1, v2):
        if v2 in self[v1] or v1 in self[v2]:
            return True
        return False

    def degree(self, nodes=None):
        if isinstance(nodes, Iterable):
            return {v: len(self[v]) for v in nodes}
        else:
            return len(self[nodes])

    def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
        """ Returns a truncated random walk.

            path_length: Length of the random walk.
            alpha: probability of restarts.
            start: the start node of the random walk.
        """
        G = self
        if start:
            path = [start]
        else:
            # Sampling is uniform w.r.t V, and not w.r.t E
            path = [rand.choice(list(G.keys()))]

        while len(path) < path_length:
            cur = path[-1]
            if len(G[cur]) > 0:
                if rand.random() >= alpha:
                    path.append(rand.choice(G[cur]))
                else:
                    path.append(path[0])
            else:
                break
        return [str(node) for node in path]


# TODO add build_walks in here

def build_random_walk_corpus(G, num_walks, walk_length, alpha=0, rand=random.Random(0)):
    """
    :param G: Graph
    :param num_walks: Number of randowm walks per node
    :param walk_length: Length of walks
    :param alpha: Restart walk probability
    :param rand: Random Generator
    :return: List of walks
    """
    walks = []
    nodes = list(G.nodes())

    for cnt in tqdm(range(num_walks), desc='Build Random Walks'):
        rand.shuffle(nodes)
        for node in nodes:
            walks.append(G.random_walk(walk_length, rand=rand, alpha=alpha, start=node))
    return walks


"""def build_random_walk_corpus(G, num_paths, path_length, alpha=0, rand=random.Random(0)):
    nodes = list(G.nodes())
    for cnt in range(num_paths):
        rand.shuffle(nodes)
        for node in nodes:
            yield G.random_walk(path_length, rand=rand, alpha=alpha, start=node)"""


def load_pd_edgelist(file_, undirected=True):
    """ Reads from a pd dataframe saved to a CSV as an edgelist """
    G = Graph()
    with open(file_) as f:
        f.readline()  # skip the first line - header
        for l in f:
            i, x, y = l.strip().split(',')
            x = int(x)
            y = int(y)
            G[x].append(y)
            if undirected:
                G[y].append(x)

    G.make_consistent()
    return G




