import unittest
from src.data_preprocessor import DataPreprocessor
import pandas as pd

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """
        Initialize the DataPreprocessor instance before each test.
        """
        self.preprocessor = DataPreprocessor(data_path='./data/dataset.csv')

    def test_load_data(self):
        """
        Test the load_data method to ensure data is loaded correctly from the CSV file.
        """
        self.preprocessor.load_data()
        # Check if the data is not None
        self.assertIsNotNone(self.preprocessor.data, "Data should not be None after loading.")
        # Check if the data is a pandas DataFrame
        self.assertIsInstance(self.preprocessor.data, pd.DataFrame, "Loaded data should be a DataFrame.")

    def test_prepare_data(self):
        """
        Test the prepare_data method to ensure it processes the data correctly.
        """
        self.preprocessor.load_data()
        self.preprocessor.prepare_data()
        # Check if 'MODULE_TEMPERATURE' column exists in the features
        self.assertIn('MODULE_TEMPERATURE', self.preprocessor.x.columns, "'MODULE_TEMPERATURE' should be in the features.")
        # Check if 'DC_POWER' column exists in the target variable
        self.assertIn('DC_POWER', self.preprocessor.y.name, "'DC_POWER' should be the target variable.")

    def test_split_data(self):
        """
        Test the split_data method to ensure the data is split correctly into training and test sets.
        """
        self.preprocessor.load_data()
        self.preprocessor.prepare_data()
        self.preprocessor.split_data()
        # Check if training and test sets are of equal length
        self.assertEqual(len(self.preprocessor.x_train), len(self.preprocessor.y_train), "Training feature and target sizes should match.")
        self.assertEqual(len(self.preprocessor.x_test), len(self.preprocessor.y_test), "Test feature and target sizes should match.")

if __name__ == '__main__':
    unittest.main()