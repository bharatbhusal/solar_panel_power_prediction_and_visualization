import matplotlib.pyplot as plt

class Visualizer:
    """
    A class to visualize the dataset using various plots.

    Attributes:
        data (DataFrame): The pandas DataFrame containing the dataset.
    """

    def __init__(self, data):
        """
        Initializes the Visualizer with the dataset.

        Parameters:
            data (DataFrame): The pandas DataFrame to be visualized.
        """
        self.data = data
        self.data_sample = self.data.iloc[:51, :]
        

    def plot_time_vs_temperature(self):
        """
        Plots a bar chart of Time vs Temperature.

        Displays the module temperature over time using a bar chart.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Time vs Temperature", fontsize=20)
        plt.xlabel("Date Time", fontsize=15)
        plt.ylabel("Module Temperature", fontsize=15)
        plt.bar(self.data_sample['DATE_TIME'], self.data_sample['MODULE_TEMPERATURE'], color="orange", lw=5)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_temperature_histogram(self):
        """
        Plots a histogram of Module Temperature.

        Displays the distribution of module temperatures using a histogram.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Histogram of Temperature", fontsize=20)
        plt.xlabel("Temperature", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.hist(self.data_sample['MODULE_TEMPERATURE'], rwidth=0.9, color='skyblue', edgecolor='black')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_time_vs_temperature_line(self):
        """
        Plots a line graph of Time vs Temperature.

        Displays the module temperature over time using a line graph.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Line Plot Graph (Date Time vs Temperature)", fontsize=20)
        plt.xlabel("Date Time", fontsize=15)
        plt.ylabel("Module Temperature", fontsize=15)
        plt.plot(self.data_sample['DATE_TIME'], self.data_sample['MODULE_TEMPERATURE'], label="Line Plot", color="red", lw=2)
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_temperature_vs_dc_power(self):
        """
        Plots a bar chart of Temperature vs DC Power.

        Displays the DC power generated based on module temperature using a bar chart.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Temperature vs DC Power", fontsize=20)
        plt.xlabel("Module Temperature", fontsize=15)
        plt.ylabel("DC Power", fontsize=15)
        plt.bar(self.data_sample['MODULE_TEMPERATURE'], self.data_sample['DC_POWER'], color="orange", lw=5)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_temperature_vs_dc_power_line(self):
        """
        Plots a line graph of Temperature vs DC Power.

        Displays the relationship between module temperature and DC power using a line graph.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Line Plot Graph (Temperature vs DC Power)", fontsize=20)
        plt.xlabel("Module Temperature", fontsize=15)
        plt.ylabel("DC Power", fontsize=15)
        plt.plot(self.data_sample['MODULE_TEMPERATURE'], self.data_sample['DC_POWER'], label="Line Plot", color="red", lw=2)
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_scatter_temperature_vs_dc_power(self):
        """
        Plots a scatter plot of Temperature vs DC Power.

        Displays the DC power against module temperature using a scatter plot.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Scatter Plot Graph", fontsize=20)
        plt.xlabel("Module Temperature", fontsize=15)
        plt.ylabel("DC Power", fontsize=15)
        plt.scatter(self.data_sample['MODULE_TEMPERATURE'], self.data_sample['DC_POWER'], label="Scatter Plot", color="blue", s=100, marker='o')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_dc_power_histogram(self):
        """
        Plots a histogram of DC Power.

        Displays the distribution of DC power generated using a histogram.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Histogram of DC Power", fontsize=20)
        plt.xlabel("DC Power", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        plt.hist(self.data_sample['DC_POWER'], rwidth=0.9, color='skyblue', edgecolor='black')
        plt.grid(True)
        plt.tight_layout()
        plt.show()