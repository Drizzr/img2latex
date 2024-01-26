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
          console.log('Model loaded');
          resolve();
          /* this.model?.load().then(() => {
            console.log('Model initialized');
            resolve();
          }); */
        })
        .catch((error) => {
          reject(error);
        });
    });
  }

  async predict() {
    if (!this.model) {
      throw new Error('Model not loaded');
    }
    tf.tidy(() => {
      try {
        const tensor = tf.zeros([1, 96, 480, 1]); //this.preprocesserService.preprocess(img);
        console.log(tensor);
        console.log(this.model?.modelVersion);

        this.model?.executeAsync(tensor).then((result) => {
          console.log(result);
        });
      } catch (error) {
        console.log(error);
      }
    });
    /*     this.preprocesserService.loadTestImage().then((img) => {

    }); */
  }
}
