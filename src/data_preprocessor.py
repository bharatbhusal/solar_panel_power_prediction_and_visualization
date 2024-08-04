import pandas as pd
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """
    A class to handle data preprocessing tasks, including loading,
    preparing, and splitting the data.

    Attributes:
        data_path (str): Path to the CSV file containing the data.
        data (pd.DataFrame): DataFrame containing the loaded data.
        x (pd.DataFrame): Features for the model.
        y (pd.Series): Target variable for the model.
        x_train (pd.DataFrame): Training features.
        x_test (pd.DataFrame): Testing features.
        y_train (pd.Series): Training target variable.
        y_test (pd.Series): Testing target variable.
    """

    def __init__(self, data_path):
        """
        Initializes the DataPreprocessor with the path to the data file.

        Parameters:
            data_path (str): Path to the CSV file containing the data.
        """
        self.data_path = data_path
        self.data = None
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        """
        Loads the data from the CSV file into a DataFrame.
        This method should be called before preparing or splitting the data.
        """
        self.data = pd.read_csv(self.data_path)
        print("Data loaded successfully")

    def prepare_data(self):
        """
        Prepares the data for model training:
        - Converts the 'DATE_TIME' column to datetime format.
        - Defines the features (x) and target (y) for the model.

        This method should be called after loading the data and before splitting the data.
        """
        # Ensure data is loaded before preparing
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Convert 'DATE_TIME' to datetime format
        self.data['DATE_TIME'] = pd.to_datetime(self.data['DATE_TIME'], dayfirst=True)

        # Define features and target
        self.x = self.data[['MODULE_TEMPERATURE']]
        self.y = self.data['DC_POWER']
        print("Data prepared")

    def split_data(self):
        """
        Splits the data into training and testing sets:
        - Features (x) and target (y) are split into training and testing datasets.

        This method should be called after preparing the data.
        """
        # Ensure data is prepared before splitting
        if self.x is None or self.y is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, train_size=0.80, random_state=1)
        print("Data split into training and testing sets")