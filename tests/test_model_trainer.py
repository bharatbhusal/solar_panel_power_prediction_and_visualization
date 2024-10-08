import unittest
import os
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

        # Define file path to save the model
        self.model_filepath = './data/test_saved_model.keras'

    def test_train_and_save_model(self):
        """
        Test the train_model method to ensure it returns a history object
        with 'loss' and 'mae' metrics containing the expected number of epochs.
        Also, save the trained model to the specified file path.
        """
        # Train the model for 5 epochs
        history = self.trainer.train_model(epochs=5)
        
        # Check if history contains the 'loss' and 'mae' metrics
        self.assertIn('loss', history.history, "The training history should contain 'loss'.")
        self.assertIn('mae', history.history, "The training history should contain 'mae'.")

        # Check if the length of 'loss' and 'mae' metrics matches the number of epochs
        self.assertEqual(len(history.history['loss']), 5, "The length of 'loss' should match the number of epochs.")
        self.assertEqual(len(history.history['mae']), 5, "The length of 'mae' should match the number of epochs.")

        # Save the trained model to the specified file path
        self.trainer.save_model(self.model_filepath)
        self.assertTrue(os.path.exists(self.model_filepath), "The model file should exist after saving.")

    def test_load_model(self):
        """
        Test the save_model and load_model methods to ensure the model is saved and loaded correctly.
        """
        try:
            # Ensure the model has been trained and saved in the previous test
            if not os.path.exists(self.model_filepath):
                self.test_train_and_save_model()  # Train and save the model if not already done

            # Load the model using the class method
            loaded_trainer = ModelTrainer.load_model(self.model_filepath, self.x_train, self.y_train)

            # Check if the loaded model's structure is the same as the original
            self.assertEqual(len(loaded_trainer.model.layers), len(self.trainer.model.layers), "The loaded model should have the same number of layers as the original model.")

        finally:
            # Clean up by removing the saved model file
            if os.path.exists(self.model_filepath):
                os.remove(self.model_filepath)

if __name__ == '__main__':
    unittest.main()