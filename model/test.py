from tensorflow import keras
import tensorflow
import numpy as np


lstm = keras.layers.LSTM(3, return_sequences=True)

inputs = np.array([[[1, 2, 3], [4, 5, 6]]], dtype="float32")



print(lstm(np.array([[[1, 2, 3]]], dtype="float32")))

print(lstm(np.array([[[4, 5, 6]]], dtype="float32")))

lstm.reset_states(states=[np.ones((1, 3)), np.ones((1, 3))])