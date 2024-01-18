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

        batch_size = enc_out.shape[0]

        #dec_states, context_t = self.model.init_decoder(enc_out)

        # first input to the decoder is <sos>

        tgt = tf.ones((batch_size, 1), dtype=tf.int32) * self.vocab.tok_to_id["<sos>"]
        embedded = self.model.embedding(tgt)
        
        
        # greedy decoding
        formula = []

        state = self.model.rnn_decoder.get_initial_state(embedded)

        for t in range(self.max_len):
            
            pred_id, state = self._get_next_token(enc_out, tgt, state)

            formula.append(self.vocab.id_to_tok[int(pred_id)])
            if pred_id == self.vocab.tok_to_id["<eos>"]:
                break
            # use the predicted token as the next input to the decoder
            tgt = tf.ones((1, 1), dtype=tf.int32) * pred_id

        return " ".join(formula)
    
    def _get_next_token(self, context, prior_token, state, temperature=1.0):
        """Get the next token"""
        # predict logit
        logit, state = self.model(context, prior_token, state, return_state=True)

        # sample token
        pred_id = tf.argmax(logit, axis=2).numpy()[0]
        
        return pred_id, state
    
    def _print_target_sequence(self, target):
        """Print the target sequence"""
        formula = []
        for i in target:
            formula.append(self.vocab.id_to_tok[int(i)])
        print("Target sequence: ", " ".join(formula))
