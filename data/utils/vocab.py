
from collections import Counter
import pandas as pd


def build_vocab(datasets=["train", "test", "validate"], min_freq=1, specials=['<pad>', '<unk>', "<eos>", "<sos>"], path="data/"):
    """
    Build a vocabulary from a list of tokens.
    Args:
        data (list): list of csv dataset-names to build the vocabulary from
        min_freq (int): minimum frequency of a token to be included in the vocabulary
        max_size (int): maximum size of the vocabulary
        specials (list): list of special tokens
    Returns:
        Vocab: a vocabulary instance
    """
    
    print("Building vocab...")
    c = Counter()
    for dataset in datasets:
        print("Reading " + dataset + " dataset...")
        df = pd.read_csv(path + "im2latex_" + dataset + ".csv")
        for formula in df["formula"]:
            c.update(str(formula).strip("\n").split()) 
    tokens = sorted([token for token, freq in c.items() if freq >= min_freq])
    tokens = specials + tokens
    print("- done. {}/{} tokens added to vocab.".format(len(tokens), len(c) + len(specials)))
    return tokens


def write_vocab(vocab, filename):
    """Writes a vocab to a file

    Writes one word per line.

    Args:
        vocab: iterable that yields word
        filename: path to vocab file

    Returns:
        write a word per line

    """
    print("Writing vocab...")
    with open(filename, "w") as f:
        for i, word in enumerate(vocab):
            if i != len(vocab) - 1:
                f.write("{}\n".format(word))
            else:
                f.write(word)
    print("- done. {} tokens".format(i+1))


class Vocabulary:

    # class to load and store vocabulary

    def __init__(self, vocab_file_path):

        self.path = vocab_file_path
        self.load_vocab()

    
    def load_vocab(self):

        self.tok_to_id = dict()
        self.id_to_tok = []
        with open(self.path) as f:
            for idx, token in enumerate(f):
                token = token.strip()
                self.tok_to_id[token] = idx
                self.id_to_tok.append(token)

        self.n_tokens = len(self.tok_to_id)
        print("Vocabulary loaded. {} tokens".format(self.n_tokens))



if __name__ == "__main__":
    # logic to build the voacabulary and save it to a file
    write_vocab(build_vocab(min_freq=10), "data/vocab.txt")