# Tensorflow.py to Tensorflow.js

TO convert a trained model from tensorflow.py to tensorflow.js, install tensorflowjs:

```bash
pip install tensorflowjs
```bash

Then save the original model as a .keras model (--save_keras) when running train.py

Then go into the folder where the model is and run (for e.g.):
  
  ```bash
  tensorflowjs_converter --input_format=keras ./Epoche26/model.keras ./Epoche26/converted_model
  ```
