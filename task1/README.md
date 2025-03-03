# Task 1
## Set up
To install all needed libraries you should run 
```
pip install -r requirements.txt
```
Actually there will be one more installation(keras tuner) but as I run it in Google collab and it is needed only for model tuning.
## Files Content
### classifiers.py
Interface class called MnistClassifierInterface, 3 classes (Random Forest, Feed-Forward Neural Network, Convolutional Neural Network) hat implements MnistClassifierInterface and MnistClassifier.
- MnistClassifierInterface  
  Interface for Random Forest, Feed-Forward Neural Network and Convolutional Neural Network
- Random Forest
  Class that implements MnistClassifierInterface. It creates and trains Random forest model for MNIST classification dataset.
- Feed-Forward Neural Network
  Class that implements MnistClassifierInterface. It creates and trains Feed-Forward Neural Network model for MNIST classification dataset.
- Convolutional Neural Network
  Class that implements MnistClassifierInterface. It creates and trains Convolutional Neural Network model for MNIST classification dataset.
### demo.ipynb
Here you can see how model works(their accuracy). Random forest - 0.97, Feed-Forward Neural Network - 0.98 and Convolutional Neural Network - 0.99. They all already are with tunrd hyperparameters.
### tuning.ipynb
Here i tried to find the best parameters for required models. More details you can see there.
