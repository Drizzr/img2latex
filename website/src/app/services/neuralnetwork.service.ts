import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';

@Injectable({
  providedIn: 'root',
})
export class NeuralnetworkService {
  constructor() {}

  async loadModel() {
    const model = await loadGraphModel('assets/model/model.json');
    return model;
  }

  async predict(model: tf.GraphModel, imgElement: any) {
    /*     const pred = tf.tidy(() => {
      // Convert the canvas pixels to
      let img = tf.browser.fromPixels(imgElement, 1);
      img = img.reshape([1, 28, 28, 1]);
      img = tf.cast(img, 'float32');

      // Make and format the predications
      const output = model.predict(img) as any;

      // Save predictions on the component
      const predictions = Array.from(output.dataSync());

      return predictions;
    });

    return pred; */
  }
}
