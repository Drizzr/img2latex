import tensorflow as tf
from tensorflow import keras


class Img2LaTex_model(keras.Model):
    def __init__(self, embedding_dim, decoder_units, vocab_size, attention_head_size=16, encoder_units=8, enc_out_dim=512 ,dropout=0.5):
        
        super().__init__()

        self.cnn_encoder = keras.Sequential([   # using sequential automatically forwars the input to the next layer
                # input size: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
                keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, W, H, 64)
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, W/2, H/2, 64)
                keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, W/2, H/2, 128)
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, W/2, H/2, 128)
                keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, W/4, W/4, 256)
                keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, W/8, H/8, 256)
                keras.layers.Conv2D(filters = enc_out_dim, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, W/8, H/8, enc_out_dim)
                keras.layers.MaxPooling2D(pool_size=(2, 1))
                # (batch_size, W/16, H/8, 256)
            ])
        # -> output shape: (batch_size, 6, 60, enc_out_dim) or more generally (batch_size, W', H', enc_out_dim)



        self.encoder_rnn  = tf.keras.layers.Bidirectional(
                                    merge_mode='sum',
                                    layer=tf.keras.layers.GRU(encoder_units,
                                    # Return the sequence
                                    return_sequences=True,
                                    recurrent_initializer='glorot_uniform'))
        
        self.dropout = keras.layers.Dropout(dropout)
        
        self.embedding = keras.layers.Embedding(vocab_size, embedding_dim)

        self.rnn_decoder = keras.layers.GRU(decoder_units, return_state=True, return_sequences=True,
                                recurrent_initializer='glorot_uniform')

        self.cross_attention = CrossAttention(units=attention_head_size, num_heads=1)

        self.output_layer = keras.layers.Dense(vocab_size, activation='softmax')

    @tf.function(reduce_retracing=True)      
    def call(self, imgs, formulas, state=None, return_state=False):
        """
        imgs: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
        
        formulas: [B, T]
        
        epsilon: float, for scheduled sampling
        
        returns: logits (B, Max_len, Vocab_size)
        """
        encoded_imgs = self.encode(imgs) # -> (batch_size, 6*60, enc_out_dim)
        
        logits, state = self.decode(encoded_imgs, formulas, state=state)

        if return_state:
            return logits, state
        return logits
    

        
    def encode(self, imgs):
        # input size: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
        x = self.cnn_encoder(imgs)

        # -> output shape: (batch_size, W', H', enc_out_dim)

        # flatten last two dimensions
        B, W, H, C = x.shape
        #x = tf.reshape(x, (B, W*H, C)) #-> (batch_size, W' * H', enc_out_dim)
        x = tf.keras.layers.Reshape((W*H, C))(x)

        x  = self.encoder_rnn(x) # -> (batch_size, W' * H', encoder_units)

        x = self.dropout(x)

        return x
    
    def decode(self, encoded_imgs, formulas, state=None):

        embeddings = self.embedding(formulas) # -> (batch_size, max_len, embedding_dim)
        
        x, state = self.rnn_decoder(embeddings, initial_state=state)

        
        x = self.cross_attention(x, encoded_imgs)

        logits = self.output_layer(x)

        return logits, state

    def build_graph(self, raw_shape):
        x = keras.Input(shape=raw_shape, batch_size=1)
        formula = keras.Input(shape=(150,), batch_size=1)
        return keras.Model(inputs=[x, formula], outputs=self.call(x, formula))
    

class CrossAttention(keras.layers.Layer):

    """
    Implements a Cross Attention block as proposed in the paper "Attention is all you need"
    along with the cross attention heads, layer normalization and residual connections
    """
    
    def __init__(self, units, num_heads=1):
        super().__init__()

        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=units)
        self.layer_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = keras.layers.Add()
    

    @tf.function
    def call(self, x, context):

        attn_output, attn_scores = self.mha(
            query=x,
            value=context,
            return_attention_scores=True)
        
        
        x = self.add([x, attn_output])
        x = self.layer_norm(x)  # residual connection
        return x 


if __name__ == "__main__":
    raw_input = (480, 96, 1)
    model = Img2LaTex_model(80, 512, 500)
    model.build_graph(raw_input).summary()


    