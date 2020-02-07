import h5py
from tqdm import tqdm

from gensim.corpora import Dictionary
from gensim.utils import strided_windows

from .graph_utils import load_pd_edgelist, build_random_walk_corpus


def create_examples(args, edgelist_path, save=True):
    n_walks, walk_len = args.get('n_walks'), args.get('walk_len')
    window_size = args.get('window_size')

    G = load_pd_edgelist(edgelist_path)
    dictionary, walks = generate_walks(G, n_walks, walk_len)

    examples = []
    for walk in tqdm(enumerate(walks), desc='Generating Examples:', total=len(walks)):
        windows = strided_windows(walk, window_size)
        for w in windows:
            center, context = w[0], w[1:]  # Add entity id as well
            examples.append([walk[0], center, context])

    if save:
        param_str = f'{n_walks}_walks_{walk_len}_walk_len_{window_size}_ws'
        example_pth = save_examples(f'graph_examples_{param_str}.h5', examples)
        dict_pth = save_dictionary(f'graph_dictionary_{param_str}', dictionary)
        return example_pth, dict_pth

    return examples, dictionary


def generate_walks(G, n_walks, walk_len):
    walks = build_random_walk_corpus(G, n_walks, walk_len)
    # Now we have a Gensim Dictionary to work with
    dictionary = Dictionary(walks)
    # Covert docs to indexes in dictionary
    return dictionary, [dictionary.doc2idx(w) for w in tqdm(walks, desc='Converting to indicies')]


def save_examples(path, examples):
    hf = h5py.File(path, 'w')
    # Save globals, i.e. doc ids to their own group
    g1 = hf.create_group('globals')
    g1.create_dataset('data', data=examples[:, 0])
    # Save centers to their own group
    g2 = hf.create_group('centers')
    g2.create_dataset('data', data=examples[:, 1])
    # Save contexts to their own group
    g3 = hf.create_group('contexts')
    g3.create_dataset('data', data=examples[:, 2:])
    # Cloase otherwise file gets messed up
    hf.close()
    return path


def save_dictionary(path, dictionary):
    dictionary.save(path)
    return path
