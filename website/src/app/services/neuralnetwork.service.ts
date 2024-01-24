import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { loadGraphModel } from '@tensorflow/tfjs-converter';
import { PreprocesserService } from './preprocesser.service';

@Injectable({
  providedIn: 'root',
})
export class NeuralnetworkService {
  model: tf.GraphModel | undefined;

  constructor(private preprocesserService: PreprocesserService) {}

  async loadModel(): Promise<void> {
    return new Promise((resolve, reject) => {
      loadGraphModel('assets/model/model.json')
        .then((model) => {
          this.model = model;
          resolve();
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  async predict() {
    this.preprocesserService.loadTestImage().then((img) => {
      const tensor = this.preprocesserService.preprocess(img);
      console.log(tensor);
      const prediction = this.model?.executeAsync(tensor).then((result) => {
        console.log(result);
      });
    });
  }
}
