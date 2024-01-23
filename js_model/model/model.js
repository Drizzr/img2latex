import * as tf from "@tensorflow/tfjs-node";

class Img2LaTexModel {
  constructor(
    embeddingDim,
    decoderUnits,
    vocabSize,
    attentionHeadSize = 16,
    encoderUnits = 8,
    encOutDim = 512,
    dropout = 0.5
  ) {
    this.cnnEncoder = tf.sequential({
      layers: [
        tf.layers.conv2d({
          filters: 64,
          kernelSize: [3, 3],
          activation: "relu",
          padding: "same",
          inputShape: [480, 96, 1],
        }),
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }),
        tf.layers.conv2d({
          filters: 128,
          kernelSize: [3, 3],
          activation: "relu",
          padding: "same",
        }),
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }),
        tf.layers.conv2d({
          filters: 256,
          kernelSize: [3, 3],
          activation: "relu",
          padding: "same",
        }),
        tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }),
        tf.layers.conv2d({
          filters: encOutDim,
          kernelSize: [3, 3],
          activation: "relu",
          padding: "same",
        }),
        tf.layers.maxPooling2d({ poolSize: [2, 1] }),
      ],
    });

    this.encoderRNN = tf.layers.bidirectional({
      layer: tf.layers.gru({
        units: encoderUnits,
        returnSequences: true,
        recurrentInitializer: "glorotUniform",
      }),
      mergeMode: "sum",
    });

    this.dropout = tf.layers.dropout({ rate: dropout });

    this.embedding = tf.layers.embedding({
      inputDim: vocabSize,
      outputDim: embeddingDim,
    });

    this.rnnDecoder = tf.layers.gru({
      units: decoderUnits,
      returnSequences: true,
      returnState: true,
      recurrentInitializer: "glorotUniform",
    });

    this.crossAttention = new CrossAttention(attentionHeadSize, 1);

    this.outputLayer = tf.layers.dense({
      units: vocabSize,
      activation: "softmax",
    });
  }

  call(imgs, formulas, state = null, returnState = false) {
    const encodedImgs = this.encode(imgs);
    const [logits, newState] = this.decode(encodedImgs, formulas, state);

    if (returnState) {
      return [logits, newState];
    }
    return logits;
  }

  encode(imgs) {
    const x = this.cnnEncoder.predict(imgs);
    const [B, W, H, C] = x.shape;
    const reshapedX = x.reshape([B, W * H, C]);

    const encodedX = this.encoderRNN.predict(reshapedX);
    const droppedX = this.dropout.predict(encodedX);

    return droppedX;
  }

  decode(encodedImgs, formulas, state = null) {
    const embeddings = this.embedding.predict(formulas);
    const [x, newState] = this.rnnDecoder.predict(embeddings, { initialState: state });

    const crossAttentionOutput = this.crossAttention.call(x, encodedImgs);
    const logits = this.outputLayer.predict(crossAttentionOutput);

    return [logits, newState];
  }

  buildGraph(rawShape) {
    const input = tf.input({ shape: rawShape, batchSize: 1 });
    const formulaInput = tf.input({ shape: [150], batchSize: 1 });
    const output = this.call(input, formulaInput);
    return tf.model({ inputs: [input, formulaInput], outputs: output });
  }
}

class CrossAttention {
  constructor(units, numHeads = 1) {
    this.mha = tf.layers.multiHeadAttention({
      numHeads: numHeads,
      keySize: units,
    });
    this.layerNorm = tf.layers.layerNormalization({ epsilon: 1e-6 });
    this.add = tf.layers.add();
  }

  call(x, context) {
    const [attnOutput] = this.mha.apply(x, context, context);
    const addedOutput = this.add.apply([x, attnOutput]);
    const normOutput = this.layerNorm.apply(addedOutput);
    return normOutput;
  }
}

// Example usage
const rawInputShape = [480, 96, 1];
const model = new Img2LaTexModel(80, 512, 500);
model.buildGraph(rawInputShape).summary();
