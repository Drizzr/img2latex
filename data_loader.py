import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd


def load_and_preprocess_img(path):
    img = Image.open(path)
    img = np.array(img)
    img = img / 255.0
    return img

def strArrayToNumpyArray(strArray):
    #takes an string in the form "[1, 2, 3]" and returns a numpy array
    strArray = strArray.replace("[", "")
    strArray = strArray.replace("]", "")
    strArray = strArray.replace(" ", "")
    return np.array(strArray.split(","), dtype=np.int32)

def create_dataset(path="data/tokenized_data/", img_path = "data/preprocessed_imgs/", type="train", batch_size=32, vocab=None):

    df = pd.read_csv(path + "im2latex_"+type+"_tokenized.csv", delimiter=";")
    img_paths = df["image_path"].values
    labels = df["formula"].values


    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    ds = ds.map(lambda x, y: (load_and_preprocess_img(img_path + x), strArrayToNumpyArray(y)))

    return ds.shuffle().batch(batch_size)



#Discarded since tokenizing is now done in the data preprocessing step
""" def tokenize_label(label, vocab, max_len): 
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
    
    return np.array(tokenize_formula) """