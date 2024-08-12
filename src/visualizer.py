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
    
    def plot_temperature_vs_dc_power_with_prediction(self, predicted_temperature, predicted_power):
        """
        Plots a line graph of Temperature vs DC Power and highlights the predicted power output.

        Displays the relationship between module temperature and DC power using a line graph,
        and marks the predicted power output for a given temperature.

        Parameters:
            predicted_temperature (float): The temperature value used for prediction.
            predicted_power (float): The predicted DC power output corresponding to the temperature.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Line Plot Graph (Temperature vs DC Power)", fontsize=20)
        plt.xlabel("Module Temperature", fontsize=15)
        plt.ylabel("DC Power", fontsize=15)
        
        # Plot the relationship between temperature and DC power
        plt.plot(self.data_sample['MODULE_TEMPERATURE'], self.data_sample['DC_POWER'], 
                label="Temperature vs DC Power", color="red", lw=2)
        
        # Highlight the predicted point
        plt.scatter(predicted_temperature, predicted_power, 
                    color='blue', s=200, marker='o', label=f"Predicted Power: {predicted_power:.2f}W")
        
        # Add the predicted point annotation
        plt.annotate(f"Predicted Power: {predicted_power:.2f}W", 
                    xy=(predicted_temperature, predicted_power), 
                    xytext=(predicted_temperature + 1, predicted_power + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=12, color='blue')
        
        plt.xticks(rotation=45, ha='right')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_scatter_temperature_vs_dc_power_with_prediction(self, predicted_temperature, predicted_power):
        """
        Plots a scatter plot of Temperature vs DC Power and highlights the predicted power output.

        Displays the relationship between module temperature and DC power using a scatter plot,
        and marks the predicted power output for a given temperature.

        Parameters:
            predicted_temperature (float): The temperature value used for prediction.
            predicted_power (float): The predicted DC power output corresponding to the temperature.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Scatter Plot (Temperature vs DC Power)", fontsize=20)
        plt.xlabel("Module Temperature", fontsize=15)
        plt.ylabel("DC Power", fontsize=15)
        
        # Scatter plot for the existing data
        plt.scatter(self.data_sample['MODULE_TEMPERATURE'], self.data_sample['DC_POWER'], 
                    label="Actual Data", color="skyblue", s=100, marker='o')
        
        # Highlight the predicted point
        plt.scatter(predicted_temperature, predicted_power, 
                    color='red', s=200, marker='*', label=f"Predicted Power: {predicted_power:.2f}W")
        
        # Add annotation for the predicted point
        plt.annotate(f"Predicted Power: {predicted_power:.2f}W", 
                    xy=(predicted_temperature, predicted_power), 
                    xytext=(predicted_temperature + 1, predicted_power + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=12, color='red')
        
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_dc_power_histogram_with_prediction(self, predicted_power):
        """
        Plots a histogram of DC Power and adds a vertical line for the predicted power output.

        Displays the distribution of DC power generated using a histogram and marks the
        predicted power output for the given temperature.

        Parameters:
            predicted_power (float): The predicted DC power output.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Histogram of DC Power with Predicted Power", fontsize=20)
        plt.xlabel("DC Power", fontsize=15)
        plt.ylabel("Frequency", fontsize=15)
        
        # Histogram for the existing data
        plt.hist(self.data_sample['DC_POWER'], rwidth=0.9, color='skyblue', edgecolor='black')
        
        # Add a vertical line for the predicted power
        plt.axvline(predicted_power, color='red', linestyle='dashed', linewidth=2, label=f"Predicted Power: {predicted_power:.2f}W")
        
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_temperature_vs_dc_power_bar_with_prediction(self, predicted_temperature, predicted_power):
        """
        Plots a bar chart of Temperature vs DC Power and highlights the predicted power output.

        Displays the DC power generated based on module temperature using a bar chart,
        and marks the predicted power output for a given temperature.

        Parameters:
            predicted_temperature (float): The temperature value used for prediction.
            predicted_power (float): The predicted DC power output corresponding to the temperature.
        """
        plt.figure(figsize=(15, 8))
        plt.title("Temperature vs DC Power with Predicted Power", fontsize=20)
        plt.xlabel("Module Temperature", fontsize=15)
        plt.ylabel("DC Power", fontsize=15)
        
        # Highlight the predicted bar
        colors = ['red' if temp == predicted_temperature else 'orange' for temp in self.data_sample['MODULE_TEMPERATURE']]
        
        # Plot the bar chart
        plt.bar(self.data_sample['MODULE_TEMPERATURE'], self.data_sample['DC_POWER'], color=colors, lw=5)
        
        # Add an annotation for the predicted point
        plt.annotate(f"Predicted Power: {predicted_power:.2f}W", 
                    xy=(predicted_temperature, predicted_power), 
                    xytext=(predicted_temperature + 1, predicted_power + 50),
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    fontsize=12, color='red')
        
        plt.xticks(rotation=45, ha='right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()