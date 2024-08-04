import unittest
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from src.model_trainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):
    def setUp(self):
        """
        Initialize the ModelTrainer instance before each test.
        """
        # Create a simple Sequential model with one input and one output layer
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_dim=1))
        self.model.add(Dense(1))
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
        
        # Create dummy training data
        self.x_train = np.array([[1], [2], [3], [4], [5]])
        self.y_train = np.array([1, 2, 3, 4, 5])
        
        # Initialize ModelTrainer with the model and training data
        self.trainer = ModelTrainer(self.model, self.x_train, self.y_train)

    def test_train_model(self):
        """
        Test the train_model method to ensure it returns a history object
        with 'loss' and 'mae' metrics containing the expected number of epochs.
        """
        # Train the model for 5 epochs
        history = self.trainer.train_model(epochs=5)
        
        # Check if history contains the 'loss' and 'mae' metrics
        self.assertIn('loss', history.history, "The training history should contain 'loss'.")
        self.assertIn('mae', history.history, "The training history should contain 'mae'.")

        # Check if the length of 'loss' and 'mae' metrics matches the number of epochs
        self.assertEqual(len(history.history['loss']), 5, "The length of 'loss' should match the number of epochs.")
        self.assertEqual(len(history.history['mae']), 5, "The length of 'mae' should match the number of epochs.")

if __name__ == '__main__':
    unittest.main()