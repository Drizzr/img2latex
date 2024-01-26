const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const fs = require("fs");

async function predict() {
  const MODEL_URL = "data/model.json";
  const handler = tfn.io.fileSystem(MODEL_URL);

  const model = await tf.loadGraphModel(handler);
  let tensor = loadImageTensor("1.png");
  let prediction = await model.predictAsync(tensor);
  console.log(prediction);
  let probabilities = softmax(predictions, -1);
}

function loadImageTensor(path) {
  const imageBuffer = fs.readFileSync(path);
  let tfimage = tfn.node.decodeImage(imageBuffer, 1); //default #channel 4
  //reshape
  tfimage = tfimage.reshape([1, 96, 480, 1]);
  tfimage = tfimage.cast("float32");
  console.log(tfimage);
  return tfimage;
}

predict();
