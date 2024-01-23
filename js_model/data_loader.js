const tf = require("@tensorflow/tfjs-node");
const fs = require("fs").promises;
const path = require("path");
const sharp = require("sharp");
const csv = require("csv-parser");

async function loadAndPreprocessImg(filePath) {
  const imgBuffer = await fs.readFile(filePath);
  const img = tf.node.decodeImage(imgBuffer, 1);
  return img.toFloat().div(tf.scalar(255.0));
}

function strArrayToNumpyArray(strArray) {
  // Takes a string in the form "[1, 2, 3]" and returns a numpy array
  const cleanStrArray = strArray.replace(/\[|\]|\s/g, "");
  const numericArray = cleanStrArray.split(",").map(Number);
  return tf.tensor1d(numericArray, "float32");
}

async function createDataset(
  dataPath = "data/tokenized_data/",
  imgPath = "data/preprocessed_imgs/",
  type = "train",
  batchSize = 32,
  vocab = null
) {
  const csvPath = path.join(dataPath, `im2latex_${type}_tokenized.csv`);
  const records = [];

  // Read CSV file
  await new Promise((resolve, reject) => {
    fs.createReadStream(csvPath)
      .pipe(csv({ delimiter: ";" }))
      .on("data", (record) => {
        records.push(record);
      })
      .on("end", () => {
        resolve();
      })
      .on("error", (error) => {
        reject(error);
      });
  });

  const imgPaths = records.map((record) => record.image_path);
  const labels = records.map((record) => record.formula);

  const ds = tf.data
    .array(imgPaths)
    .mapAsync(async (imgPath, index) => {
      const img = await loadAndPreprocessImg(path.join(imgPath, imgPath));
      const label = strArrayToNumpyArray(labels[index]);
      return [img, label];
    })
    .shuffle(32000)
    .batch(batchSize);

  return ds;
}

// Example usage
const dataset = createDataset();
dataset.forEachAsync((batch) => {
  console.log("Batch:", batch);
});
