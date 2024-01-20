import tensorflow as tf
from model import Img2LaTex_model, Trainer, LatexProducer
import json
from data.utils import Vocabulary
from data_loader import create_dataset
import argparse
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True


PATH = "checkpoints/chechpoint_epoch_28_0.0%_estimated_loss_0.16"

# load params
parser = argparse.ArgumentParser(description="Play with the model.")
parser.add_argument("--render", action='store_true',
                        default=False, help="Render formulas")
args = parser.parse_args()

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

if args.render:
    for imgs, formulas in dataset:
        print("_______________________________________________________________________________________________")
        gen._print_target_sequence(tf.squeeze(formulas).numpy())
        generatedSequence = gen._greedy_decoding(imgs)
        print("greedy_search: ", generatedSequence)
        print("_______________________________________________________________________________________________")
        fig, ax = plt.subplots(figsize=(6, 4), tight_layout=True)
        ax.set_title(r"$\displaystyle " + generatedSequence + "$", fontsize=20)
        ax.imshow(tf.squeeze(imgs).numpy(), cmap="gray")
        
        plt.show()
else:
    for imgs, formulas in dataset:
        print("_______________________________________________________________________________________________")
        gen._print_target_sequence(tf.squeeze(formulas).numpy())
        print("beam_search: ", gen._beam_search(imgs, beam_width=10))
        print("greedy_search: ", gen._greedy_decoding(imgs))
        print("_______________________________________________________________________________________________")
        
