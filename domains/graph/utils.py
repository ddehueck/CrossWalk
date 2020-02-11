import h5py
import os.path
import numpy as np

from tqdm import tqdm
from gensim.corpora import Dictionary
from gensim.utils import strided_windows

from .graph import build_random_walk_corpus


def load_examples(args, edgelist_path, G):
    # Unpack params
    n_walks, walk_len = args.get('n_walks'), args.get('walk_len')
    window_size = args.get('window_size')

    # Filenames for examples to be saved to
    param_str = f'{n_walks}_walks_{walk_len}_walk_len_{window_size}_ws'
    example_pth = f'data/graph_examples_{param_str}.h5'
    dict_pth = f'data/graph_dictionary_{param_str}.gensim'

    if os.path.isfile(example_pth) and os.path.isfile(dict_pth):
        print(f'Loading examples from: {example_pth}')
        print(f'Loading dictionary from: {dict_pth}')
        return example_pth, dict_pth

    # Generate randomwalks
    dictionary, walks = generate_walks(G, n_walks, walk_len)

    # Create Examples
    examples = []
    for walk in tqdm(walks, desc='Generating Examples:', total=len(walks)):
        windows = strided_windows(walk, window_size)
        for w in windows:
            center, context = w[0], w[1:]  # Add entity id as well
             # convert to global entity ids!
            _global = int(dictionary[walk[0]])
            _center = int(dictionary[center])
            _context = np.array([int(dictionary[c]) for c in context])
            # save example
            examples.append([_global, _center, _context])

    # Save Examples!
    save_examples(example_pth, examples)
    save_dictionary(dict_pth, dictionary)
    return example_pth, dict_pth


def generate_walks(G, n_walks, walk_len):
    walks = build_random_walk_corpus(G, n_walks, walk_len)
    # Now we have a Gensim Dictionary to work with
    dictionary = Dictionary(walks)
    # Covert docs to indexes in dictionary
    return dictionary, [dictionary.doc2idx(w) for w in tqdm(walks, desc='Converting to indicies')]


def save_examples(path, examples):
    if os.path.isfile(path):
        return path

    examples = np.array(examples)
    hf = h5py.File(path, 'w')
    # Save globals, i.e. doc ids to their own group
    g1 = hf.create_group('globals')
    g1.create_dataset('data', data=examples[:, 0].tolist())
    # Save centers to their own group
    g2 = hf.create_group('centers')
    g2.create_dataset('data', data=examples[:, 1].tolist())
    # Save contexts to their own group
    g3 = hf.create_group('contexts')
    g3.create_dataset('data', data=examples[:, 2].tolist())
    # Cloase otherwise file gets messed up
    hf.close()
    return path


def save_dictionary(path, dictionary):
    if os.path.isfile(path):
        return path

    dictionary.save(path)
    return path
