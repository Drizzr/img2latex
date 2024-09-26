import tensorflow as tf
import numpy as np

class LatexProducer():

    def __init__(self, model, vocab, max_len=150):
        self.model = model
        
        self.vocab = vocab

        self.max_len = max_len
    

    def _greedy_decoding(self, img):
        """Greedy Decoding"""
        # encode

        tgt = tf.ones((1, 1), dtype=tf.int32) * self.vocab.tok_to_id["<sos>"]

        encoded_img = self.model.encode(img)
        
        # greedy decoding
        formula = []

        state = None

        for t in range(self.max_len):
            
            pred_id, state = self._get_next_token(encoded_img, tgt, state)
            
            if pred_id == self.vocab.tok_to_id["<eos>"]:
                break

            formula.append(self.vocab.id_to_tok[int(pred_id)])
            # use the predicted token as the next input to the decoder
            tgt = tf.ones((1, 1), dtype=tf.int32) * pred_id

        return " ".join(formula)
    

    def _beam_search(self, img, beam_width=3):

        # set up initial state for decoder

        k = beam_width

        tgt = tf.ones((1, 1), dtype=tf.int32) * self.vocab.tok_to_id["<sos>"]

        logit = self.model(img, tgt, return_state=False, training=False) # logit shape (beam_width, 1, vocab_size)

        log_probs = tf.math.log(logit)
        log_probs = tf.squeeze(log_probs) 

        topk_log_probs, topk_ids = tf.math.top_k(log_probs, k=k) # topk_log_probs shape (beam_width, k), topk_ids shape (beam_width, k)

        beams = tf.reshape(topk_ids, (beam_width, 1)) # beams shape (beam_width, 1)

        beam_scores = tf.reshape(topk_log_probs, (beam_width, 1)) # beam_scores shape (beam_width, 1)

        encoded_img = self.model.encode(img) # encoded_shape (1, 360, 64) 
        encoded_img = tf.broadcast_to(encoded_img, [beam_width, encoded_img.shape[1], encoded_img.shape[2]])       

        finished_beams = []
        finished_beams_scores = []


        for t in range(self.max_len):

            encoded_img = encoded_img[:beams.shape[0], :, :]

            logit, _ = self.model.decode(encoded_img, beams) # logit shape (beam_width, 1, vocab_size)

            
            logit = logit[:, -1 , :] # logit shape (beam_width, vocab_size)

            log_probs = tf.math.log(logit) # log_probs shape (beam_width, 1, vocab_size)
            log_probs = tf.reshape(log_probs, (k, -1)) # log_probs shape (beam_width, vocab_size)

            log_probs = tf.reshape(log_probs, (k * self.vocab.n_tokens)) # log_probs shape (beam_width * vocab_size,)

            topk_log_probs, topk_ids = tf.math.top_k(log_probs, k=k) # topk_log_probs shape (beam_width, k), topk_ids shape (beam_width, k)

            topk_log_probs = topk_log_probs / (t+1)**beam_width # normalize by length
            
            topk_beam_index = topk_ids // self.vocab.n_tokens # topk_beam_index shape (beam_width, k)
            topk_ids = topk_ids % self.vocab.n_tokens # topk_ids shape (k)
            
            beams = tf.stack([beams[index, :] for index in topk_beam_index], axis=0) # beams shape (beam_width, t+1)
            beam_scores = tf.stack([beam_scores[index, :] for index in topk_beam_index], axis=0) # beam_scores shape (beam_width, t+1)
    
            topk_ids = tf.reshape(topk_ids, (k, 1)) # topk_ids shape (beam_width, 1)
            topk_log_probs = tf.reshape(topk_log_probs, (k, 1)) # topk_log_probs shape (beam_width, 1)

            beams = tf.concat([beams, topk_ids], axis=1) # beams shape (beam_width, t+2)
            beam_scores += topk_log_probs # beam_scores shape (beam_width, t+2)


            # remove finished beams

            for index in range(k-1, -1, -1):
                if beams[index, -1] == self.vocab.tok_to_id["<eos>"]:

                    finished_beams.append(beams[index, :].numpy().tolist())
                    finished_beams_scores.append(float(beam_scores[index, :].numpy()))
                    if index == 0:
                        beams = beams[1:, :]
                        beam_scores = beam_scores[1:, :]
                    elif index == k:
                        beams = beams[:k, :]
                        beam_scores = beam_scores[:k, :]
                    else:
                        beams = tf.concat([beams[:index, :], beams[index+1:, :]], axis=0)
                        beam_scores = tf.concat([beam_scores[:index, :], beam_scores[index+1:, :]], axis=0)

                    k -= 1
                    img = img[:k, :, :, :]
                    # reshape img

                    

            if k == 0:
                break

        
        if not finished_beams:
            for i in range(k):
                finished_beams.append(beams[i, :].numpy().tolist())
                finished_beams_scores.append(float(beam_scores[i, :].numpy()))


        # select the best beam
        
        
        best_beam_index = int(np.argmax(finished_beams_scores))

        
        best_beam = finished_beams[best_beam_index]

        formula = []
        for i in best_beam:

            if i == self.vocab.tok_to_id["<eos>"]:
                break
            formula.append(self.vocab.id_to_tok[int(i)])

        return " ".join(formula)


        
    def _get_next_token(self, encoded_img, prior_token, state, temperature=1.0):
        """Get the next token"""
        # predict logit

        logit, state = self.model.decode(encoded_img, prior_token, state)
        
        # sample token
        pred_id = tf.argmax(logit, axis=2).numpy()[0]
        
        return pred_id, state
    
    def _print_target_sequence(self, target):
        """Print the target sequence"""
        formula = []
        for i in target:
            if i == self.vocab.tok_to_id["<eos>"]:
                break
            formula.append(self.vocab.id_to_tok[int(i)])
        
        formula.pop(0)
        print("Target sequence: ", " ".join(formula))
