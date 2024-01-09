import argparse
from data.utils.vocab import Vocabulary
from data_loader import create_dataset
from model import Img2LaTex_model, Trainer
import tensorflow as tf


def build_model(model):
    # generate input to call method
    x = tf.random.uniform((1, 480, 96, 1))
    formula = tf.random.uniform((1, 150))

    return model

def main():

    # get args
    parser = argparse.ArgumentParser(description="Train the model.")

    # model args
    parser.add_argument("--embedding_dim", type=int, default=80)

    parser.add_argument("--lstm_rnn_h", type=int, default=512, help="size of the lstm hidden state")

    parser.add_argument("--enc_out_dim", type=int, default=512, help="size of the encoder output")

    parser.add_argument("--max_len", type=int, default=512, help="size of the token length")

    parser.add_argument("--dropout", type=float, default=0.5, help="dropout rate")

    parser.add_argument("--num_epochs", type=int, default=15, help="number of epochs")

    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--print_freq", type=int, default=100)

    parser.add_argument("--sample_method", type=str, default="teacher_forcing",
                        choices=('teacher_forcing', 'exp', 'inv_sigmoid'),
                        help="The method to schedule sampling")
    
    parser.add_argument("--decay_k", type=float, default=0.002,)


    parser.add_argument("--from_check_point", action='store_true',
                        default=False, help="Training from checkpoint or not")
    
    parser.add_argument("--clip", type=float, default=5.0, help="gradient clipping")

    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")

    parser.add_argument("--lr_decay", type=float, default=0.5, help="learning rate decay")

    

    args = parser.parse_args()

    from_check_point = args.from_check_point
    
    if from_check_point:
        pass # implement load from checkpoint

    print("training args: ", args)

    # load vocab
    vocab = Vocabulary("data/vocab.txt")

    vocab_size = vocab.n_tokens

    dataset = create_dataset(batch_size=args.batch_size, vocab=vocab, type="train", max_len=args.max_len)
    

    print("construct dataset...")

    model = Img2LaTex_model(args.embedding_dim, args.lstm_rnn_h, vocab_size, args.enc_out_dim, args.max_len, args.dropout)

    model = build_model(model)

    print("model built...")

    trainer = Trainer(model, dataset, args)

    trainer.train()



if __name__ == "__main__":
    main()




