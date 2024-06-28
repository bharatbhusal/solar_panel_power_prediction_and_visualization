# Solar Panel Power Prediction and Visualization

Welcome to the Solar Panel Power Prediction and Visualization project repository! This project focuses on predicting the power output of solar panels based on module temperature data using machine learning techniques. Here's a detailed guide on setting up the environment, running the project, visualizing data, evaluating the model, and more.

## Overview

The project includes data preprocessing, model training using TensorFlow/Keras, evaluation using metrics like accuracy and a confusion matrix, and visualization of various aspects of the dataset.

## Project Structure

- `dataset.csv`: Dataset containing solar panel data including module temperature and DC power.
- `solar_power_prediction.py`: Python script for data preprocessing, model creation, training, and prediction.
- `README.md`: This file, providing an overview of the project, setup instructions, and usage details.
- `requirements.txt`: Python dependencies required to run the project.
- `.gitignore`: Specifies files and directories that should be ignored by Git.

## Setting Up Your Environment

### Prerequisites

Ensure you have Python 3.6 or higher installed on your system along with the Pip package manager.

### Setting up the codebase

Using a virtual environment is recommended to manage dependencies and avoid conflicts with other projects. Follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/bharatbhusal/solar_panel_power_prediction_and_visualization.git
   cd solar_panel_power_prediction_and_visualization
   ```

2. **Create the Virtual Environment:**

   ```bash
   python -m venv venv
   ```

   This command creates a virtual environment named `venv` in the current directory.

3. **Activate the Virtual Environment:**

   - On **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - On **macOS and Linux**:

     ```bash
     source venv/bin/activate
     ```

   When the virtual environment is activated, your command prompt will change to show the name of the virtual environment, typically something like `(venv)` before the prompt.

4. **Install Dependencies:**

   With the virtual environment activated, install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

   This will install TensorFlow, pandas, matplotlib, and scikit-learn, along with other necessary dependencies.

5. **Run Your Script:**

   Execute your Python script while the virtual environment is active:

   ```bash
   python solar_power_prediction.py
   ```

6. **Deactivate the Virtual Environment:**

   After you're done working in the virtual environment, deactivate it:

   ```bash
   deactivate
   ```

   This command deactivates the virtual environment and returns you to the system's default Python interpreter.

## Running the Project

1. **Prepare the Dataset:**

   Ensure that `dataset.csv` is in the project directory. This file contains the solar panel data needed for training and predictions.

2. **Execute the Script:**

   Run the prediction script to start the process:

   ```bash
   python solar_power_prediction.py
   ```

3. **Input Temperature Values:**

   Follow the prompts in the script to input temperature values. The model will use these inputs to predict the corresponding power output.

## Data Visualization

Explore various plots generated using matplotlib in the `solar_power_prediction.py` script to visualize different aspects of the dataset:

- Time vs. Temperature
- Histogram of Temperature
- Line Plot (Date Time vs. Temperature)
- Temperature vs. DC Power
- Line Plot (Temperature vs. DC Power)
- Scatter Plot (Temperature vs. DC Power)
- Histogram of DC Power

## Model Evaluation

The model's performance is evaluated using metrics such as accuracy and a confusion matrix, providing insights into its effectiveness.

## Future Improvements

- Explore more complex neural network architectures.
- Enhance data preprocessing techniques.
- Deploy the model as a service for real-time predictions.

## Contributing

Contributions to the project are welcome! If you have suggestions for improvements or find issues, please open an issue or submit a pull request on GitHub.
