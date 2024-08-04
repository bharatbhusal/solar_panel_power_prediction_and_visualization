import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.predictor import Predictor

class TestPredictor(unittest.TestCase):
    def setUp(self):
        """
        Initialize the Predictor instance before each test.
        """
        # Create and compile a simple Sequential model
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=1))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

        # Train the model with dummy data
        self.model.fit(np.array([[1], [2], [3], [4], [5]]), np.array([1, 2, 3, 4, 5]), epochs=5, verbose=0)
        
        # Initialize Predictor with the trained model
        self.predictor = Predictor(self.model)

    def test_predict(self):
        """
        Test the predict method of the Predictor class to ensure it returns a numeric power value.
        """
        # Test prediction with a sample temperature value
        power = self.predictor.predict(25.0)
        
        # Check if the returned power value is of type float
        self.assertIsInstance(power, np.float32, "The returned power value should be a float.")

if __name__ == '__main__':
    unittest.main()