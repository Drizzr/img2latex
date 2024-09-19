
# Img2LaTeX

## Overview

This repository contains the code for training and deploying a deep learning model that converts images of LaTeX formulas into corresponding LaTeX code. The model is designed with a hybrid architecture that combines **CNNs**, **Bidirectional GRUs**, and **Cross-Attention** mechanisms. This architecture ensures that the model can focus on specific regions of the image while generating the LaTeX sequence.

---

## Model Architecture

The **Img2LaTeX** model follows a Transformer-inspired design, with RNNs replacing the self-attention layers. Here's a more detailed breakdown of the architecture:

### 1. **Encoder (CNN + Bidirectional GRU)**

- **CNN Block**: The grayscale image is processed by a Convolutional Neural Network (CNN) to extract visual features, reducing dimensionality while capturing important patterns in LaTeX formulas (e.g., symbols, lines, curves).
- **Bidirectional GRU**: The extracted features are then passed through a Bidirectional GRU. This layer processes the image features in both forward and backward directions, allowing the model to capture global context across the image. The output is a sequence of encoded image features.

### 2. **Decoder (Unidirectional GRU)**

- **Unidirectional GRU**: The decoder processes token embeddings sequentially using a unidirectional GRU. This ensures that only past tokens influence the current prediction, simulating an autoregressive model.
- **Input**: Token embeddings and the attention-weighted image features are used as input at each decoding step to predict the next token in the LaTeX sequence.

### 3. **Cross-Attention Mechanism**

The **cross-attention** mechanism is the core of the architecture, dynamically aligning the encoded image features with the current hidden state of the decoder. This allows the decoder to focus on relevant parts of the image during each step of LaTeX generation.

- **Query**: The hidden state of the decoder.
- **Key & Value**: The encoded image features.

### 4. **Prediction Layer**

The cross-attention output helps the decoder generate a probability distribution over the vocabulary of LaTeX tokens. The most probable token is selected (greedy decoding) or beam search is used for more complex decoding. This process continues until the end-of-sequence token (`<eos>`) is generated.

### 5. **Tokenizer and Vocabulary**

The input LaTeX sequences are tokenized using a predefined vocabulary, and all sequences are padded to a uniform length with `<pad>`, `<sos>` (start of sequence), and `<eos>` (end of sequence) tokens. This ensures consistent input dimensions during training and inference.

---

## Training Workflow

The model is trained to predict LaTeX tokens from images of formulas. The following steps outline the training process:

### Dataset

- **Img2LaTeX100k**: The dataset used consists of images of LaTeX formulas and their corresponding LaTeX code.
- **Data Preprocessing**: 
  - Filter formulas exceeding the max token length (150 tokens).
  - Resize all images to a uniform size of `[96, 480]` to maintain consistent input dimensions.

### Training Configuration

1. **Command**: You can initiate training with the following command:
   ```bash
   python train.py --embedding_dim 80 --num_epochs 40 --batch_size 120 --lr 0.0003
   ```

2. **Hyperparameters**:
   - `embedding_dim`: Dimensionality of the token embeddings.
   - `decoder_units`: Number of units in the GRU decoder.
   - `enc_out_dim`: Dimensionality of the encoder output.
   - `attention_head_size`: Size of the attention heads.
   - `max_len`: Maximum length of the LaTeX sequence.
   - `dropout`: Dropout rate for regularization.
   - `num_epochs`: Number of training epochs.
   - `batch_size`: Batch size for training.
   - `lr`: Learning rate for optimization.
   - `save_dir`: Directory where model checkpoints are saved.

### Hardware and Training Time

- **Hardware**: AMD Ryzen 5 3600X, 16GB RAM.
- **Training Time**: Approximately 50 hours for 40 epochs, with a batch size of 120.
- **Loss**: Final loss of around 0.15 with ~70% token accuracy on the validation set. Inference time per formula is approximately 2 seconds using greedy search.

---

## Inference Workflow (Sampling)

Once the model is trained, it can generate LaTeX code from formula images using the following script:

### Inference Command

```bash
python sample.py --render
```

### Output

- If `--render` is specified, the generated LaTeX formula is displayed alongside the input image using `matplotlib`.
- Without `--render`, the script outputs the LaTeX code to the console.

---

Here’s a styled and expanded version of the **Examples** section, along with some additional context:


## Examples

Below are a few examples of the **Img2LaTeX** model in action, showcasing its ability to convert images of LaTeX formulas into their corresponding LaTeX code (top image shows the rendered formula based on the model output). While the model performs well on the training and validation datasets, it may struggle with unseen images or those that differ significantly from the dataset it was trained on.

### Example 1: Formula from the Validation Set
In this case, the model successfully recognizes the structure of the LaTeX formula and generates an accurate output.

![Validation Formula](https://github.com/user-attachments/assets/d20f5be0-ba69-4208-9fda-d21c7cfa62e3)


---

### Example 2: Complex LaTeX Structure
This example shows the model’s capability to handle more complex LaTeX formulas, demonstrating its effectiveness in generating multi-step expressions.

![Complex Formula](https://github.com/user-attachments/assets/937e8ff9-15a4-4887-815c-3798a40a2a88)


---

### Example 3: Issues with Unseen Data
When tested with images outside the training/validation dataset, the model may struggle with accuracy, as seen in this example. The formula produced here does not perfectly match the expected LaTeX sequence.

![Unseen Data Formula](https://github.com/user-attachments/assets/b96a7422-137e-4fa7-a693-2bb43829c360)


---

### Example 4: Partial Recognition
The model can sometimes partially recognize formulas but make errors with symbols or structure, especially when faced with noisy or unfamiliar inputs. This can be improved with better data diversity and model refinement.

![Partial Formula Recognition](https://github.com/user-attachments/assets/52b964e3-c0c3-4676-b2e7-7e352b5ef78d)


---

### Performance and Challenges
While the **Img2LaTeX** model demonstrates strong performance on familiar data, it still faces challenges with generalizing to images outside the training set. The complexity of LaTeX structures and image variability can affect accuracy, especially in cases where the input image is noisy or the formula involves rarely seen symbols.

Further work on data augmentation, model fine-tuning, and improvements in attention mechanisms could help the model generalize better to unseen examples.


---

## Key Functions

### 1. **build_model(model, formula_len)**

- Builds and initializes the model by providing random input for both image features and LaTeX sequences. Prints the model build time.

### 2. **load_from_checkpoint(save_dir, args, vocab, dataset, val_dataset)**

- Loads a model from saved checkpoints, including hyperparameters and weights. Adjusts the model to continue training from the last saved epoch.

### 3. **Trainer.train()**

- Handles the training loop, including calculating loss, updating weights, and saving checkpoints.

### 4. **LatexProducer**

- Generates LaTeX sequences during inference using either greedy decoding or beam search. It also supports rendering of the predicted formulas.
- For a smaller, standalone repository focused just on inference, you can check out the [Img2Latex Generator](https://github.com/Drizzr/Img2Latex_generator).

---

## Future Work and Improvements

- **Optimizing Training Data**: Improve the diversity of the dataset to prevent overfitting.
- **Browser Compatibility**: Adapt the model for browser-based use with TensorFlow.js.
- **Overfitting Solutions**: Address overfitting by improving data regularization techniques and experimenting with PyTorch for more efficient memory management.

---

## Requirements
Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Conclusion

The **Img2LaTeX** model offers a robust approach for converting LaTeX formula images into LaTeX code. With its hybrid CNN-RNN structure and cross-attention mechanism, the model captures both spatial and sequential information effectively. The model performs well on the **Img2LaTeX100k** dataset, and future improvements aim to expand its usability in browser-based applications.

```

