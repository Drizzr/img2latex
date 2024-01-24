#plots the loss of all epochs during training
import matplotlib.pyplot as plt
import os
import argparse
import json
import matplotlib as mpl


"""epochs = []
losses = []

vocab = Vocabulary("data/vocab.txt")
vocab_size = vocab.n_tokens

val_data_set = create_dataset(vocab=vocab, batch_size=1, type="validate")

for folder in folders:
    
    with open(args.path + "/" + folder + "/params.json", "r") as f:
        params = json.load(f)


    model = Img2LaTex_model(embedding_dim=params["embedding_dim"], enc_out_dim=params["enc_out_dim"], vocab_size=vocab_size,
                        attention_head_size=params["attention_head_size"], encoder_units=params["encoder_units"],
                        decoder_units=params["decoder_units"],)

    model.load_weights(args.path + "/" + folder + "/weights.h5")

    trainer = Trainer(model, dataset=None, val_dataset=val_data_set, epochs=1, batch_size=1, vocab=vocab, max_len=150, checkpoint_path=None)

    val_los = trainer._validate()

    losses.append(val_los)

    epochs.append(params["epoch"])
"""

with open("data/losses.json", "r") as f:
    losses = json.load(f)



mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['text.usetex'] = True
mpl.rcParams["text.latex.preamble"] = r'\usepackage{siunitx}'

plt.plot(losses)


plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()
