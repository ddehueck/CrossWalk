import re
import h5py
import pandas as pd
from tqdm import tqdm

from gensim.utils import strided_windows
from gensim.corpora import Dictionary


def create_examples(args, df_path, save=True):
    window_size = args.get('window_size')
    dictionary, walks = tokenize_files(df_path)

    examples = []
    for i, walk in tqdm(enumerate(walks), desc='Generating Examples:', total=len(walks)):
        windows = strided_windows(walk, window_size)
        for w in windows:
            center, context = w[0], w[1:]  # Add entity id as well
            examples.append([i, center, context])

    if save:
        param_str = f'{window_size}_ws'
        example_pth = save_examples(f'graph_examples_{param_str}.h5', examples)
        dict_pth = save_dictionary(f'graph_dictionary_{param_str}', dictionary)
        return example_pth, dict_pth

    return examples, dictionary


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
    # Close otherwise file gets messed up
    hf.close()
    return path


def save_dictionary(path, dictionary):
    dictionary.save(path)
    return path


def tokenize_files(df_path):
    data = pd.read_csv(df_path, na_filter=False)['language'].values
    cleaned_docs = [tokenize_doc(f) for f in tqdm(data, desc='Tokenizing Docs')]
    # Now we have a Gensim Dictionary to work with
    dictionary = Dictionary(cleaned_docs)
    # Remove any tokens with a frequency less than 10
    dictionary.filter_extremes(no_below=10, no_above=0.75)
    # Covert docs to indexes
    indexed_docs = [dictionary.doc2idx(d) for d in tqdm(cleaned_docs, desc='Converting to indicies')]
    # Remove out of vocab tokens
    return dictionary, [[t for t in d if t != -1] for d in tqdm(indexed_docs, desc="Removing out-of-vocab tokens")]


def tokenize_doc(doc):
    # Convert to text make lowercase
    clean_doc = [t.lower().strip() for t in doc.split()]
    # Only allow characters in the alphabet, '_', and digits
    clean_doc = [re.sub('[^a-zA-Z0-9]', '', t) for t in clean_doc]
    # Remove any resulting empty indices
    return [t for t in clean_doc if len(t) > 0]