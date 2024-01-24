import { Component } from '@angular/core';
import { NeuralnetworkService } from './services/neuralnetwork.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss'],
})
export class AppComponent {
  title = 'website';

  constructor(private neuralnetworkService: NeuralnetworkService) {
    this.neuralnetworkService.loadModel().then(() => {
      this.neuralnetworkService.predict();
    });
  }
}
