import tensorflow as tf
from model import Img2LaTex_model, LatexProducer, create_dataset
import json
from data.utils import Vocabulary
import numpy as np


vocab = Vocabulary("data/vocab.txt")
vocab_size = vocab.n_tokens

val_data_set = create_dataset(vocab=vocab, batch_size=1, type="validate")


    
with open("checkpoints/chechpoint_epoch_41_0.0%_estimated_loss_0.287" +  "/params.json", "r") as f:
    params = json.load(f)


model = Img2LaTex_model(embedding_dim=params["embedding_dim"], enc_out_dim=params["enc_out_dim"], vocab_size=vocab_size,
                    attention_head_size=params["attention_head_size"], encoder_units=params["encoder_units"],
                    decoder_units=params["decoder_units"],)

img = tf.random.uniform((1, 96, 480, 1))
formula = tf.random.uniform((1, 1))
model(img, formula)

model.load_weights("checkpoints/chechpoint_epoch_41_0.0%_estimated_loss_0.287" + "/weights.h5")

total = len(val_data_set) * 151
correct = 0
state = None
j = 0

for imgs, formulas in val_data_set:
    
    j += 1
    tgt = tf.ones((1, 1), dtype=tf.int32) * vocab.tok_to_id["<sos>"]

    for i in range(151):
        logits, state = model(imgs, tgt, state, return_state=True, training=False)
        pred = tf.argmax(logits, axis=2).numpy()[0]
        if pred == formulas[0][i+1].numpy():
            correct += 1
        tgt = tf.ones((1, 1), dtype=tf.int32) * pred

        if pred == vocab.tok_to_id["<eos>"]:
            
            padd_list = np.array([vocab.tok_to_id["<pad>"]] * (151 - i))

            correct += np.sum(padd_list == formulas[0][i+1:])

            break

            

    if j % 10 == 0:
        print("Validation accuracy: ", correct/(j*151))
        print(f"Step {j} of {total/151}")
print("Validation accuracy: ", correct/total)