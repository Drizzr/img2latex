import tensorflow as tf


class Encoder(tf.keras.layers.Layer):

    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()
        self.embedding_dim = embedding_dim

        
        self.conv1 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), activation='relu', padding='same')
        
        self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), activation='relu', padding='same')
        self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2)
        self.conv3 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same')
        self.conv4 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), activation='relu', padding='same')
        self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 1))

        # flatten last two dimensions
        self.flatten = tf.keras.layers.Reshape((-1, 256))
        self.dense1 = tf.keras.layers.Dense(256, activation='relu')
        self.dense2 = tf.keras.layers.Dense(embedding_dim, activation='relu')

        # bidirectional lstm
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True, return_state=True))

        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool3(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)

        x = self.dropout(x)

        output, forward_h, forward_c, backward_h, backward_c = self.lstm(x)

        state_h = tf.keras.layers.Concatenate()([forward_h, backward_h])
        state_c = tf.keras.layers.Concatenate()([forward_c, backward_c])

        return output, state_h, state_c







    
