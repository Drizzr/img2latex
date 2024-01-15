import tensorflow as tf

class LatexProducer():

    def __init__(self, model, vocab, max_len=150):
        self.model = model
        
        self.vocab = vocab

        self.max_len = max_len
    

    def predict(self, img):
        # implement beam search
        return self._greedy_decoding(img)
    

    def _greedy_decoding(self, img):
        """Greedy Decoding"""
        # encode
        enc_out = self.model.encode(img)

        dec_states, context_t = self.model.init_decoder(enc_out)

        # first input to the decoder is <sos>

        tgt = tf.ones((1, 1), dtype=tf.int32) * self.vocab.tok_to_id["<sos>"]

        # greedy decoding
        formula = []
        for t in range(self.max_len):
            dec_states, context_t, logit = self.model.step_decoding(
                dec_states, context_t, enc_out, tgt)

            # predict token
            pred_id = tf.argmax(logit, axis=1).numpy()[0]
            formula.append(self.vocab.id_to_tok[int(pred_id)])
            if pred_id == self.vocab.tok_to_id["<eos>"]:
                break
            # use the predicted token as the next input to the decoder
            tgt = tf.ones((1, 1), dtype=tf.int32) * pred_id

        return " ".join(formula)