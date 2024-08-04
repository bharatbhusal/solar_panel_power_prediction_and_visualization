class ModelEvaluator:
    """
    A class to evaluate the performance of a trained neural network model.

    Attributes:
        model (Sequential): The Keras Sequential model to be evaluated.
        x_test (DataFrame or ndarray): Features of the test dataset.
        y_test (Series or ndarray): Target values of the test dataset.
    """

    def __init__(self, model, x_test, y_test):
        """
        Initializes the ModelEvaluator with the model and test data.

        Parameters:
            model (Sequential): The trained Keras model to be evaluated.
            x_test (DataFrame or ndarray): Features of the test dataset.
            y_test (Series or ndarray): Target values of the test dataset.
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test

    def evaluate_model(self):
        """
        Evaluates the model on the test data and prints the predictions.

        The method makes predictions using the test data and rounds them to the nearest integer.
        It prints the predictions and returns them.

        Returns:
            ndarray: The rounded predictions on the test dataset.
        """
        # Predict the target values for the test set
        y_pred = self.model.predict(self.x_test).round()

        # Print the predictions
        print("Predictions on test data:")
        print(y_pred)
        
        # Return the predictions
        return y_pred