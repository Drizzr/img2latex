import tensorflow as tf
from model import Img2LaTex_model, Trainer, LatexProducer
import json
from data.utils import Vocabulary
from data_loader import create_dataset


with open("checkpoints/params.json", "r") as f:
    params = json.load(f)

vocab = Vocabulary("data/vocab.txt")

vocab_size = vocab.n_tokens

model = Img2LaTex_model(embedding_dim=params["embedding_dim"], enc_out_dim=params["enc_out_dim"], vocab_size=vocab_size,
                        attention_head_size=params["attention_head_size"], encoder_units=params["encoder_units"],
                        decoder_units=params["decoder_units"],)

x = tf.random.uniform((1, 480, 96, 1))
formula = tf.random.uniform((1, 150))
model(x, formula)

model.load_weights("checkpoints/test.h5")

