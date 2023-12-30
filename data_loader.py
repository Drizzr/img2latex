import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd


def load_and_preprocess_img(path):
    img = Image.open(path)
    img = np.array(img)
    img = img / 255.0
    return img

def tokenize_label(label, vocab, max_len):
    pad = vocab.token_to_id["<pad>"]
    sos = vocab.token_to_id["<sos>"]
    eos = vocab.token_to_id["<eos>"]
    unk = vocab.token_to_id["<unk>"]

    formula = str(label).strip("\n").split()
    tokenize_formula = []
    tokenize_formula.append(sos)
    for token in formula:
        if token in vocab.tok_to_id:
            tokenize_formula.append(vocab.token_to_id[token])
        else:
            tokenize_formula.append(unk)
    tokenize_formula.append(eos)
    for _ in range(max_len - len(formula)):
        tokenize_formula.append(pad)
    
    return np.array(tokenize_formula)


def create_dataset(path="data/", img_path = "data/processed_imgs/", type="train", batch_size=32, max_len=150, vocab=None):

    df = pd.read_csv("data/im2latex_"+type+".csv")
    img_paths = df["img_path"].values
    labels = df["formula"].values


    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(lambda x, y: (load_and_preprocess_img(img_path + x), tokenize_label(y, vocab, max_len)))

    return ds.batch(batch_size)

