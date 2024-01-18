import os
import tensorflow as tf
import math
import numpy as np
import time
from .inference import  LatexProducer

class Trainer(object):
    def __init__(self, model,
                 dataset, args, val_dataset, producer,
                 init_epoch=1, last_epoch=15):


        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.args = args

        self.step = 0
        self.total_step = 0
        self.epoch = init_epoch
        self.last_epoch = last_epoch
        self.best_val_loss = 1e18
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)

        self.losses = []

        self.generate = producer

    def train(self):
        stepComputeTime = time.process_time()
        mes = "Epoch {}, step:{}/{} {:.2f}%, Loss:{:.4f}, Perplexity:{:.4f}, time (s): {:.2f}, Epochtime (h): {:.2f}"
        print("Num GPUs Available 👀: ", len(tf.config.list_physical_devices('GPU')))
        while self.epoch <= self.last_epoch:
            losses = 0.0

            for imgs, target in self.dataset:
                
                tgt4training = target[:, :-1] # remove <eos>
                tgt4cal_loss = target[:, 1:] # remove <sos>

                epsilon = self.cal_epsilon(
                self.args.decay_k, self.total_step, self.args.sample_method)


                with tf.GradientTape() as tape:
                    logits = self.model(imgs, tgt4training, training=True) # -> (batch_size, max_len, vocab_size)
                    tgt4cal_loss = tf.one_hot(tf.cast(tgt4cal_loss, dtype=tf.int32), axis=-1, depth=logits.shape[-1])

                    # calculate loss
                    step_loss = self.loss_fn(tgt4cal_loss, logits)
                    grads = tape.gradient(step_loss, self.model.trainable_variables)

                    if self.args.clip > 0:
                        grads, _ = tf.clip_by_global_norm(grads, self.args.clip)
                    self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

                    self.step += 1
                    self.total_step += 1
                    

                
                losses += step_loss

                # log message
                if self.step % self.args.print_freq == 0:
                    avg_loss = losses / self.args.print_freq
                    remaining_time = (time.process_time() - stepComputeTime) * (len(self.dataset)-self.step)/3600
                    
                    print(mes.format(
                        self.epoch, self.step, len(self.dataset),
                        100 * self.step / len(self.dataset),
                        avg_loss,
                        2**avg_loss,
                        time.process_time() - stepComputeTime,
                        remaining_time,

                    ))
                    stepComputeTime = time.process_time()
                    self.losses.append(avg_loss)
                    losses = 0.0
                    
                    if self.step % 300 == 0:
                        print("prediceted: ")
                        print(self.generate._greedy_decoding(tf.expand_dims(imgs[0, :, : , :], 0)))
                        print(self.generate._print_target_sequence(tgt4training[0, :]))

            # one epoch Finished, calcute val loss
            val_loss = self.validate()

            self.save_model('ckpt-{}-{:.4f}'.format(self.epoch, val_loss))
            self.epoch += 1
            self.step = 0

    def validate(self):

        val_total_loss = 0.0
        mes = "Epoch {}, validation average loss:{:.4f}, Perplexity:{:.4f}"
        for imgs, target in self.val_dataset:
            imgs = imgs
            tgt4training = target[:, :-1] # remove <eos>
            tgt4cal_loss = target[:, 1:] # remove <sos>
            
            logits = self.model(imgs, tgt4training, training=False)
            tgt4cal_loss = tf.one_hot(tf.cast(tgt4cal_loss, dtype=tf.int32), axis=-1, depth=logits.shape[-1])
            loss = self.loss_fn(tgt4cal_loss, logits)
            val_total_loss += loss
        avg_loss = val_total_loss / len(self.val_dataset)
        print(mes.format(
            self.epoch, avg_loss, 2**avg_loss
        ))
        return avg_loss

    def save_model(self, model_name):

        save_path = "self.args.save_dir" + "model_name" + '.pt'
        print("Saving checkpoint to {}".format(save_path))

        self.model.save(save_path)
    

    @staticmethod
    def cal_epsilon(decay_k, step, sample_method):
        """
        Reference:
            Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks
            See details in https://arxiv.org/pdf/1506.03099.pdf
        """
        if sample_method == 'exp':
            return decay_k ** step
        elif sample_method == 'inv_sigmoid':
            return decay_k / (decay_k + math.exp(step / decay_k))
        elif sample_method == "teacher_forcing":
            return 1
        else:
            raise ValueError('Not valid sample method')