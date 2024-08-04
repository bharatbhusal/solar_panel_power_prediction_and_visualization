import unittest
from src.model_builder import ModelBuilder
from tensorflow.keras.models import Sequential

class TestModelBuilder(unittest.TestCase):
    def setUp(self):
        """
        Initialize the ModelBuilder instance before each test.
        """
        self.builder = ModelBuilder()

    def test_build_model(self):
        """
        Test the build_model method to ensure the model is built correctly.
        """
        self.builder.build_model()
        # Check if the model is an instance of Sequential
        self.assertIsInstance(self.builder.model, Sequential, "The built model should be an instance of Sequential.")
        # Check if the model has the expected number of layers
        self.assertEqual(len(self.builder.model.layers), 7, "The model should have exactly 7 layers.")

    def test_compile_model(self):
        """
        Test the compile_model method to ensure the model is compiled correctly.
        """
        self.builder.build_model()
        self.builder.compile_model()
        # Check if the optimizer is set
        self.assertIsNotNone(self.builder.model.optimizer, "The model should have an optimizer set after compilation.")
        # Check if the loss function is set correctly
        self.assertEqual(self.builder.model.loss, 'mean_squared_error', "The model should use 'mean_squared_error' as the loss function.")
        # Check if 'loss' is included in the metrics
        self.assertIn('loss', self.builder.model.metrics_names, "'loss' should be included in the model's metrics.")

if __name__ == '__main__':
    unittest.main()