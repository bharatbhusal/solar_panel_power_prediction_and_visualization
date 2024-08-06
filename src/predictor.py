import numpy as np

class Predictor:
    """
    A class to make predictions using a trained neural network model.

    Attributes:
        model (Sequential): The Keras Sequential model used for predictions.
    """

    def __init__(self, model):
        """
        Initializes the Predictor with the trained model.

        Parameters:
            model (Sequential): The Keras model to use for making predictions.
        """
        self.model = model

    def predict(self, temperature):
        """
        Predicts the power output of the solar panel for a given temperature.

        The method takes a temperature value, formats it for the model, and uses the model to predict the power output.

        Parameters:
            temperature (float): The temperature value for which to predict the power output.

        Returns:
            float: The predicted power output for the given temperature.
        """
        # Convert the temperature to a numpy array with the correct shape
        b = np.array([[temperature]])
        
        # Use the model to make a prediction
        y_pred = self.model.predict(b).round()

        # Extract the predicted power from the result
        power = y_pred[0, 0]

        # Print and return the predicted power
        print(f"The expected Power from the solar panel for the corresponding temperature is: {power} watts.")
        return power