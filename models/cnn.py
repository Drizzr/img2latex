import tensorflow as tf
from tensorflow import keras

class CNN(keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, max_len, num_filters, kernel_size, output_dim, dropout_rate):
        super(CNN, self).__init__()
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_len)
        self.conv = keras.layers.Conv1D(filters=num_filters, kernel_size=kernel_size, padding="valid", activation="relu")
        self.dropout = keras.layers.Dropout(dropout_rate)
        self.pool = keras.layers.GlobalMaxPooling1D()
        self.dense = keras.layers.Dense(output_dim, activation="softmax")
    
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.conv(x)
        x = self.dropout(x)
        x = self.pool(x)
        x = self.dense(x)
        return x