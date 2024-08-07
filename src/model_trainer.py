import os
from tensorflow.keras.models import load_model

class ModelTrainer:
    """
    A class to train a neural network model.

    Attributes:
        model (Sequential): The Keras Sequential model to be trained.
        x_train (DataFrame or ndarray): Features of the training dataset.
        y_train (Series or ndarray): Target values of the training dataset.
    """

    def __init__(self, model, x_train, y_train):
        """
        Initializes the ModelTrainer with the model and training data.

        Parameters:
            model (Sequential): The Keras model to be trained.
            x_train (DataFrame or ndarray): Features of the training dataset.
            y_train (Series or ndarray): Target values of the training dataset.
        """
        self.model = model
        self.x_train = x_train
        self.y_train = y_train

    def train_model(self, epochs=5000):
        """
        Trains the model using the provided training data.

        The method fits the model to the training data for a specified number of epochs.

        Parameters:
            epochs (int): The number of epochs to train the model. Default is 5000.

        Returns:
            History: The training history object containing details about the training process.
        """
        # Train the model on the training data
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=1)

        # Return the training history
        return history

    def save_model(self, filepath):
        """
        Saves the trained model to the specified file path using the native Keras format.

        Parameters:
            filepath (str): The file path to save the trained model. Should end with '.keras'.
        """
        if not filepath.endswith('.keras'):
            raise ValueError("The file path must end with '.keras'.")
        self.model.save(filepath)

    @classmethod
    def load_model(cls, filepath, x_train, y_train):
        """
        Loads a model from the specified file path and returns an instance of ModelTrainer.

        Parameters:
            filepath (str): The file path from which to load the model. Should end with '.keras'.
            x_train (DataFrame or ndarray): Features of the training dataset.
            y_train (Series or ndarray): Target values of the training dataset.

        Returns:
            ModelTrainer: An instance of ModelTrainer with the loaded model and training data.
        """
        if not filepath.endswith('.keras'):
            raise ValueError("The file path must end with '.keras'.")
        model = load_model(filepath, compile=False)
        return cls(model, x_train, y_train)