import tensorflow as tf
from model import Img2LaTex_model, Trainer, LatexProducer
import json
from data.utils import Vocabulary
from data_loader import create_dataset



model = tf.keras.models.load_model("checkpoints/test.keras")

