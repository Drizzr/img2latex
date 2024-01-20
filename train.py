import argparse
from data.utils.vocab import Vocabulary
from data_loader import create_dataset
from model import Img2LaTex_model, Trainer, LatexProducer
import tensorflow as tf
import time
import sys
import json
import os

def build_model(model, formula_len):
    # generate input to call method
    start_time = time.time()
    x = tf.random.uniform((1, 480, 96, 1))
    formula = tf.random.uniform((1, formula_len))
    model(x, formula)
    print("successfully built model...")
    print("time to build model: ", round(time.time() - start_time, 4), "s")
    return model


def load_from_checkpoint(save_dir, args, vocab, dataset, val_dataset):
    """Load model from checkpoint"""
    print("loading model from checkpoint...")
    with open(os.path.join(save_dir, "params.json"), "r") as f:
        params = json.load(f)

    vocab_size = vocab.n_tokens

    model = Img2LaTex_model(embedding_dim=params["embedding_dim"], enc_out_dim=params["enc_out_dim"], vocab_size=vocab_size,
                            attention_head_size=params["attention_head_size"], encoder_units=params["encoder_units"],
                            decoder_units=params["decoder_units"],)
    
    
    model = build_model(model, 152)

    model.load_weights(os.path.join(save_dir, "weights.h5"))
    print("model loaded successfully...")

    trainer = Trainer(model, dataset, args, val_dataset, init_epoch=params["epoch"], vocab_size=vocab_size, last_epoch=args.num_epochs)

    current_batch_size = args.batch_size
    checkpoint_batch_size = params["batch_size"]


    trainer.step = params["step"] * (checkpoint_batch_size // current_batch_size)

    return trainer, model
    

def main():

    # get args
    parser = argparse.ArgumentParser(description="Train the model.")

    # model args
    parser.add_argument("--embedding_dim", type=int, default=80)

    parser.add_argument("--decoder_units", type=int, default=64, help="size of the lstm hidden state")

    parser.add_argument("--enc_out_dim", type=int, default=64, help="size of the encoder output")

    parser.add_argument("--encoder_units", type=int, default=64, help="size of the lstm hidden state")

    parser.add_argument("--attention_head_size", type=int, default=32, help="size of the lstm hidden state")

    parser.add_argument("--max_len", type=int, default=512, help="size of the token length")

    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")

    parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")

    parser.add_argument("--batch_size", type=int, default=12)

    parser.add_argument("--print_freq", type=int, default=1)

    parser.add_argument("--sample_method", type=str, default="exp",
                        choices=('exp', 'inv_sigmoid', "teacher_forcing"),
                        help="The method to schedule sampling")
    
    parser.add_argument("--decay_k", type=float, default=0.002,) 


    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")
    
    parser.add_argument("--clip", type=float, default=5.0, help="gradient clipping")

    parser.add_argument("--lr", type=float, default=3*10**(-4), help="learning rate")

    parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")

    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="directory to save model")

    args = parser.parse_args()

    from_check_point = args.from_check_point
    
    
    print("_________________________________________________________________")
    print("HYPERPARAMETERS: ")
    for arg in vars(args):
        print(arg,": ", getattr(args, arg))
    print("_________________________________________________________________")
    # load vocab
    vocab = Vocabulary("data/vocab.txt")

    vocab_size = vocab.n_tokens

    dataset = create_dataset(batch_size=args.batch_size, vocab=vocab, type="train")

    val_dataset = create_dataset(batch_size=args.batch_size, vocab=vocab, type="validate")

    """for element in dataset2:
        if element[1].shape[1] != 152:
            print("found one")
    
    print(vocab_size)"""
        
    

    print("sucessfully loaded dataset...")

    if from_check_point:
        trainer, model = load_from_checkpoint(args.save_dir, args, vocab, dataset, val_dataset)
    
    else:
        model = Img2LaTex_model(args.embedding_dim, enc_out_dim=args.enc_out_dim, vocab_size=vocab_size, 
                                attention_head_size=args.attention_head_size, encoder_units=args.encoder_units,
                                decoder_units=args.decoder_units, dropout=args.dropout)

        model = build_model(model, 152)

        trainer = Trainer(model, dataset, args, val_dataset, vocab_size)
    
    print("_________________________________________________________________")
    print("MODEL SUMMARY 🍆: ")
    model.summary()

    print("begin training... 🥵")
    print(f"expected loss for random guessing: {-tf.math.log(1/vocab_size)}")
    print("_________________________________________________________________")

    try:
        trainer.train()
    except KeyboardInterrupt as e:
        print(e)
        
        trainer.save_model()
        
        sys.exit()




if __name__ == "__main__":
    main()




