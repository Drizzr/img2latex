import tensorflow as tf


class Img2LaTex_model(tf.keras.Model):
    def __init__(self, embedding_dim, dec_lstm_h, vocab_size, enc_out_dim=512 ,dropout=0.5):
        

        self.cnn_encoder = tf.Keras.sequential([   # using sequential automatically forwars the input to the next layer
                # input size: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
                tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 480, 96, 64)
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, 240, 48, 64)
                tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 240, 48, 128)
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, 120, 24, 128)
                tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 120, 24, 256)
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
                # (batch_size, 60, 12, 256)
                tf.keras.layers.Conv2D(filters = enc_out_dim, kernel_size = (3, 3), activation='relu', padding='same'),
                # (batch_size, 60, 12, 256)
                tf.keras.layers.MaxPooling2D(pool_size=(2, 1))
                # (batch_size, 30, 12, 256)
            ])
        # -> output shape: (batch_size, 60, 24, enc_out_dim) or more generally (batch_size, W', H', enc_out_dim)



        #self.bi_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True))

        self.dropout = tf.keras.layers.Dropout(dropout)
        
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        self.rnn_decoder = tf.keras.layers.LSTM(dec_lstm_h, return_sequences=True, return_state=True)

        # bandhau attention
        self.W1 = tf.keras.layers.Dense(dec_lstm_h)
        self.W2 = tf.keras.layers.Dense(dec_lstm_h)
        self.V = tf.keras.layers.Dense(1)

        self.W_out = tf.keras.layers.Dense(vocab_size, activation="softmax")

        # init decoder states
        self.init_wh = tf.keras.layers.Dense(dec_lstm_h, activation="tanh")
        self.init_wc = tf.keras.layers.Dense(dec_lstm_h, activation="tanh")
        self.init_w_context = tf.keras.layers.Dense(dec_lstm_h, activation="tanh")
        
    def forward(self, imgs, formulas, epsilon=1.0):
        """
        imgs: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
        
        formulas: [B, T]
        
        epsilon: float, for scheduled sampling
        
        returns: logits (B, Max_len, Vocab_size)
        """
        
        encoded_imgs = self.encode(imgs) # -> (batch_size, 360, 512)
        dec_states, context_t = self.init_decoder(encoded_imgs) # calc hidden States and attention for the first time step
        max_len = formulas.size(1) # this enables us to only consider the max_token length for each batch and not globaly
        logits = []

        for t in range(max_len):
            tgt = formulas[:, t:t+1] # -> (batch_size, 1)

            # we only use teacher for a subset of the data and for the rest we use the predicted token, this is called scheduled sampling and can be adjusted via epsilon
            if logits and tf.random.uniform((), 0, 1) > epsilon:
                tgt = tf.argmax(tf.math.log(logits[-1]), dim=1, keepdim=True)

        # initialize hidden and cell states of lstm
            dec_states, context_t, logit = self.step_decoding(
                dec_states, context_t, encoded_imgs, tgt)
            logits.append(logit)

            logits = tf.stack(logits, dim=1) # -> (batch_size, max_len, vocab_size)

            return logits
    
    def step_decoding(self, dec_states, context, enc_out, tgt):
        """Runing one step decoding"""
        """
        dec_states: tuple of (h_t, c_t) ergo the hidden and cell states of the decoder lstm
        enc_out: output of the encoder 
        tgt: pervious target token (either ground truth or predicted)
        o_t: attention generated from the previous step

        """
        embedded = self.embedding(tgt) # -> (batch_size, 1, embedding_dim)

        prev_y = tf.squeeze(embedded)  # -> (batch_size, embedding_dim)

        inp = tf.concat([prev_y, context], dim=1)  # -> (B, emb_size+dec_lstm_h)
        h_t, c_t = self.rnn_decoder(inp, dec_states)  # h_t:[B, dec_lstm_h]
        h_t = self.dropout(h_t)
        c_t = self.dropout(c_t) # prevent overfitting


        new_context = self._get_attn(enc_out, h_t) # -> (batch_size, dec_lstm_h)

        # calculate logit
        logit = self.W_out(h_t)

        return (h_t, c_t), new_context, logit

        
    
    def encode(self, imgs):
        # input size: [B, W, H, C] in our case [Batch_size, 480, 96, 1]
        x = self.cnn_encoder(imgs)
        # -> output shape: (batch_size, 30, 12, 256) or more generally (batch_size, W', H', 256)

        # flatten last two dimensions
        B, W, H, C = x.shape
        x = self.keras.layers.Reshape((B, W*H, C))(x)
        # -> output shape: (batch_size, W' * H' , enc_out_dim)

        #x = self.bi_lstm(x)
        # -> output shape: (batch_size, 360, 2*enc_out_dim)

        x = self.dropout(x)
        # -> output shape: (batch_size, 360, enc_out_dim)
        return x
    

    def _get_attn(self, enc_out, h_t):

        
        hidden_with_time_axis = tf.expand_dims(h_t, 1) # -> (batch_size, 1, dec_lstm_h)

        enc_out = self.W1(enc_out) # -> (batch_size, 360, dec_lstm_h)
        hidden_with_time_axis = self.W2(hidden_with_time_axis) # -> (batch_size, 1, dec_lstm_h)

        # attention_hidden_layer shape == (batch_size, 360, dec_lstm_h)
        alpha = tf.nn.tanh(enc_out + hidden_with_time_axis)

        # score shape == (batch_size, 360, 1)
        alpha = self.V(alpha)
        attention_weights = tf.nn.softmax(alpha, axis=1)

        context_vector = attention_weights * enc_out # -> (batch_size, 360, dec_lstm_h)


        context_vector = tf.reduce_sum(context_vector, axis=1) # -> (batch_size, dec_lstm_h)

    

        return context_vector, attention_weights
    
    def init_decoder(self, enc_out):
        """args:
            enc_out: the output of row encoder [B, H*W, C]
          return:
            h_0, c_0:  h_0 and c_0's shape: [B, dec_rnn_h]
            init_O : the average of enc_out  [B, dec_rnn_h]
            for decoder
        """
        mean_enc_out = enc_out.mean(dim=1)
        h = self._init_h(mean_enc_out)
        c = self._init_c(mean_enc_out)
        init_context = self._init_o(mean_enc_out)
        return (h, c), init_context

