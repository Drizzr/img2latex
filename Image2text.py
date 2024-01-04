import tensorflow as tf
from models.encoder import Encoder
from models.decoder import Decoder

class Img2LaTex(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Img2LaTex, self).__init__()
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(embedding_dim, units, vocab_size)

    def call(self, x):
        features = self.encoder(x)
        output = self.decoder(features)
        return output
    

model = Img2LaTex(256, 512, 10000)
model.summary()