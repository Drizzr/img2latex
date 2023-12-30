from tensorflow import keras
import tensorflow
import numpy as np


lstm = keras.layers.LSTM(3, return_sequences=True)

inputs = np.array([[[1, 2, 3], [4, 5, 6]]], dtype="float32")



print(lstm(inputs))

print(lstm(lstm(np.array([[[1, 2, 3]]], dtype="float32"))))
