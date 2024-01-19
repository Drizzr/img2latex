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

        tgt = tf.ones((1, 1), dtype=tf.int32) * self.vocab.tok_to_id["<sos>"]
        
        # greedy decoding
        formula = []

        state = None

        for t in range(self.max_len):
            
            pred_id, state = self._get_next_token(img, tgt, state)
            
            if pred_id == self.vocab.tok_to_id["<eos>"]:
                break

            formula.append(self.vocab.id_to_tok[int(pred_id)])
            # use the predicted token as the next input to the decoder
            tgt = tf.ones((1, 1), dtype=tf.int32) * pred_id

        return " ".join(formula)
    
    def _get_next_token(self, img, prior_token, state, temperature=1.0):
        """Get the next token"""
        # predict logit
        logit, state = self.model(img, prior_token, state, return_state=True, training=False)
        
        # sample token
        pred_id = tf.argmax(logit, axis=2).numpy()[0]
        
        return pred_id, state
    
    def _print_target_sequence(self, target):
        """Print the target sequence"""
        formula = []
        for i in target:
            formula.append(self.vocab.id_to_tok[int(i)])
        print("Target sequence: ", " ".join(formula))
