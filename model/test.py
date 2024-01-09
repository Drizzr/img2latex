from tensorflow import keras
import tensorflow as tf
import numpy as np


lstm = keras.layers.LSTM(3, return_sequences=True)

conv = keras.Sequential([   # using sequential automatically forwars the input to the next layer
                # input size: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
                keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 480, 96, 64)
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, 240, 48, 64)
                keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 240, 48, 128)
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, 120, 24, 128)
                keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 120, 24, 256)
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, 60, 12, 256)
                keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 60, 12, 256)
                keras.layers.MaxPooling2D(pool_size=(2, 1))
                # (batch_size, 30, 12, 256)
            ])

input = tf.ones((1, 480, 96, 1))
output = conv(input)
B, W, H, C = output.shape
enc_out = tf.reshape(output, (B, W*H, C))

print(enc_out.shape)
init_wh = keras.layers.Dense(512, activation="tanh")
init_wc = keras.layers.Dense(512, activation="tanh")
init_w_context = keras.layers.Dense(512, activation="tanh")

mean_enc_out = tf.math.reduce_mean(enc_out, 1) 
h = init_wh(mean_enc_out)
c = init_wc(mean_enc_out)
w_context = init_w_context(mean_enc_out)

tgt = tf.ones((1, 1))

embedding = keras.layers.Embedding(100, 512)

embedding_tgt = embedding(tgt)
prev_y = tf.squeeze(embedding_tgt, 1)

inp = tf.concat([prev_y, w_context], axis=1)

decoded = keras.layers.LSTMCell(512)(inp, (h, c))


