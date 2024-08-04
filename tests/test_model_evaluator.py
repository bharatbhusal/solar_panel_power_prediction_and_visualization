import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.model_evaluator import ModelEvaluator

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        """
        Initialize the ModelEvaluator instance before each test.
        """
        # Create a simple Sequential model with one input and one output layer
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=1))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        
        # Create dummy test data
        self.x_test = np.array([[1], [2], [3], [4], [5]])
        self.y_test = np.array([1, 2, 3, 4, 5])
        
        # Initialize the ModelEvaluator with the model and test data
        self.evaluator = ModelEvaluator(self.model, self.x_test, self.y_test)

    def test_evaluate_model(self):
        """
        Test the evaluate_model method to ensure it returns predictions of the correct length.
        """
        # Get predictions from the model
        y_pred = self.evaluator.evaluate_model()
        
        # Assert that the number of predictions matches the number of test samples
        self.assertEqual(len(y_pred), len(self.x_test), "The number of predictions should match the number of test samples.")

if __name__ == '__main__':
    unittest.main()