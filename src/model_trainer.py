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

    def train_model(self, epochs=200):
        """
        Trains the model using the provided training data.

        The method fits the model to the training data for a specified number of epochs.

        Parameters:
            epochs (int): The number of epochs to train the model. Default is 200.

        Returns:
            History: The training history object containing details about the training process.
        """
        # Train the model on the training data
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, verbose=1)

        # Return the training history
        return history