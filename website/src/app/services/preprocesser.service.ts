import { Injectable } from '@angular/core';
import * as tf from '@tensorflow/tfjs';
import { environment } from 'src/enviroments/enviroment';

@Injectable({
  providedIn: 'root',
})
export class PreprocesserService {
  constructor() {}

  loadTestImage(): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = 'assets/test_Data/1a0a2bdae8.png';
    });
  }

  preprocess(img: HTMLImageElement): tf.Tensor {
    const tensor = tf.browser
      .fromPixels(img, 1)
      .resizeNearestNeighbor(environment.input_dimensions as [number, number])
      .toFloat()
      .expandDims();
    return tensor;
  }
}
