import tensorflow as tf
from tensorflow import keras
from model.model import Img2LaTex_model
from data.utils import Vocabulary
import tensorflowjs as tfjs


"""
A Wrapper class to export the model to a tensorflow serving model.
"""

class Export(tf.Module):

    def __init__(self, model):
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 96, 480, 1], dtype=tf.float32)])
    def generate(self, imgs, max_len=150):

        """
        Args: 
            imgs: a tensor of shape (1, 96, 480, 1)
            max_len: maximum length of the generated formula
        Returns:
            a tensor of shape (max_len,) containing the ids of the generated formula
        """

        tgt = tf.ones((1, 1), dtype=tf.int32) * 3
        
        # greedy decoding
        formula = []

        state = None

        for i in range(max_len):
            logit, state = self.model(imgs, tgt, state, training=False, return_state=True)

            pred_id = int(tf.squeeze(tf.argmax(logit, axis=2)))

            tgt = tf.ones((1, 1), dtype=tf.int32) * pred_id
            
            formula.append(pred_id)

        return tf.stack(formula, axis=0)
        
    
if __name__ == "__main__":


    model = Img2LaTex_model(embedding_dim=80, enc_out_dim=64, vocab_size=485, \
                            attention_head_size=32, encoder_units=64, \
                            decoder_units=64, dropout=0.5)
    
    x = tf.random.uniform((1, 96, 480, 1))
    formula = tf.random.uniform((1, 1))
    model(x, formula, None)

    model.load_weights("checkpoints/chechpoint_epoch_41_0.0%_estimated_loss_0.287/weights.h5")
    
    export = Export(model)

    tf.saved_model.save(export, 'Img2Latex_exported',
                    signatures={'serving_default': export.generate})
    
    tfjs.converters.convert_tf_saved_model('Img2Latex_exported', 'Img2Latex_js_exported', control_flow_v2=True)
    
    """model = tf.saved_model.load("Img2Latex")

    print(model.generate(tf.random.uniform((1, 96, 480, 1))))"""