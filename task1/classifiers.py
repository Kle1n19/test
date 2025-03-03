import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from abc import ABC, abstractmethod

#Interafce
class MnistClassifierInterface(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass

# Random Forest Classifier
class RandomForestMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = RandomForestClassifier(bootstrap=False, n_estimators=200, n_jobs=-1) #Initializing model with params from tuning.ipynb

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train) #Trains the model that I create

    def predict(self, X_test):
        return self.model.predict(X_test) #Makes predictions

# FeedForward Neural Network Classifier
class FeedForwardMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = models.Sequential([
            layers.Dense(192, activation='relu', input_shape=(28*28,)),
            layers.Dropout(0.1),
            layers.Dense(96, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy']) #Initializing model with params from tuning.ipynb

    def train(self, X_train, y_train, epochs=10, batch_size=64):
        y_train = to_categorical(y_train, 10)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1) #Trains created model

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1) #Makes predictions

# CNN Classifier
class CNNMnistClassifier(MnistClassifierInterface):
    def __init__(self):
        self.model = models.Sequential([
            layers.Reshape((28, 28, 1), input_shape=(28*28,)),
            layers.Conv2D(filters=96, kernel_size=3, activation='relu'),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(filters=128, kernel_size=5, activation='relu'),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(192, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy']) #Initializing model with params from tuning.ipynb

    def train(self, X_train, y_train, epochs=4, batch_size=64):
        y_train = to_categorical(y_train, 10)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, verbose=1) #Trains created model

    def predict(self, X_test):
        return np.argmax(self.model.predict(X_test), axis=1) #Makes predictions

#MnistClassifier
class MnistClassifier:
    """
    Takes reshaped 28x28 images into 1D vectors (784 features) with normalized pixel values to [0, 1] by dividing by 255 for improved training stability.
    """

    def __init__(self, algorithm): #Chooses desired algorithm
        if algorithm == "rf":
            self.classifier = RandomForestMnistClassifier()
        elif algorithm == "nn":
            self.classifier = FeedForwardMnistClassifier()
        elif algorithm == "cnn":
            self.classifier = CNNMnistClassifier()
        else:
            raise ValueError("Invalid input. Options: 'cnn', 'rf', 'nn'.")

    def train(self, X_train, y_train):
        self.classifier.train(X_train, y_train) #Trains it

    def predict(self, X_test):
        return self.classifier.predict(X_test) #Makes predictions
