const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");

async function predict() {
  const MODEL_URL = "data/model.json";
  const handler = tfn.io.fileSystem(MODEL_URL);

  const model = await tf.loadGraphModel(handler);
  let tensor = tf.zeros([1, 96, 480, 1]);
  let prediction = model.predictAsync(tensor);
  prediction.then((res) => {
    let data = res.dataSync();
    console.log(data);
  });
}

predict();
