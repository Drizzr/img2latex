import tensorflow as tf
from model import Img2LaTex_model, Trainer, LatexProducer
import json
from data.utils import Vocabulary
from data_loader import create_dataset


PATH = "checkpoints/chechpoint_epoch_16_0.0%_estimated_loss_0.347"

with open(PATH + "/params.json", "r") as f:
    params = json.load(f)

vocab = Vocabulary("data/vocab.txt")

vocab_size = vocab.n_tokens

model = Img2LaTex_model(embedding_dim=params["embedding_dim"], enc_out_dim=params["enc_out_dim"], vocab_size=vocab_size,
                        attention_head_size=params["attention_head_size"], encoder_units=params["encoder_units"],
                        decoder_units=params["decoder_units"],)

x = tf.random.uniform((1, 480, 96, 1))
formula = tf.random.uniform((1, 150))
model(x, formula)

model.load_weights(PATH + "/weights.h5")



dataset = create_dataset(vocab=vocab, batch_size=1, type="validate")

gen = LatexProducer(model, vocab, max_len=150)

for imgs, formulas in dataset:
    print("_______________________________________________________________________________________________")
    gen._print_target_sequence(tf.squeeze(formulas).numpy())
    print("beam_search: ", gen._beam_search(imgs))
    print("greedy_search: ", gen._greedy_decoding(imgs))
    print("_______________________________________________________________________________________________")

    
