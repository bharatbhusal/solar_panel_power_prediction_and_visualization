import unittest
import pandas as pd
from src.visualizer import Visualizer

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        """
        Initialize the Visualizer instance before each test.
        """
        # Create a DataFrame with sample data for testing
        self.data = pd.DataFrame({
            'DATE_TIME': pd.date_range(start='1/1/2022', periods=50, freq='h'),
            'MODULE_TEMPERATURE': range(50),
            'DC_POWER': range(50)
        })
        # Initialize the Visualizer with the sample data
        self.visualizer = Visualizer(self.data)

    def test_plot_time_vs_temperature(self):
        """
        Test the plot for time vs. temperature.
        """
        try:
            self.visualizer.plot_time_vs_temperature()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot Time vs Temperature.")

    def test_plot_temperature_histogram(self):
        """
        Test the histogram plot for temperature.
        """
        try:
            self.visualizer.plot_temperature_histogram()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot Temperature Histogram.")

    def test_plot_time_vs_temperature_line(self):
        """
        Test the line plot for time vs. temperature.
        """
        try:
            self.visualizer.plot_time_vs_temperature_line()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot Time vs Temperature Line.")

    def test_plot_temperature_vs_dc_power(self):
        """
        Test the plot for temperature vs. DC power using a bar plot.
        """
        try:
            self.visualizer.plot_temperature_vs_dc_power()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot Temperature vs DC Power.")

    def test_plot_temperature_vs_dc_power_line(self):
        """
        Test the line plot for temperature vs. DC power.
        """
        try:
            self.visualizer.plot_temperature_vs_dc_power_line()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot Temperature vs DC Power Line.")

    def test_plot_scatter_temperature_vs_dc_power(self):
        """
        Test the scatter plot for temperature vs. DC power.
        """
        try:
            self.visualizer.plot_scatter_temperature_vs_dc_power()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot Scatter Temperature vs DC Power.")

    def test_plot_dc_power_histogram(self):
        """
        Test the histogram plot for DC power.
        """
        try:
            self.visualizer.plot_dc_power_histogram()
            success = True
        except Exception as e:
            success = False
        self.assertTrue(success, "Failed to plot DC Power Histogram.")

if __name__ == '__main__':
    unittest.main()