from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

class ModelBuilder:
    """
    A class to build and compile a neural network model for regression.

    Attributes:
        model (Sequential): The Keras Sequential model.
    """

    def __init__(self):
        """
        Initializes the ModelBuilder with an empty model.
        """
        self.model = None

    def build_model(self):
        """
        Builds the neural network model using Keras' Sequential API.

        The model architecture includes:
        - Input layer with shape (1,) (for single feature input).
        - Several hidden layers with ReLU activation function.
        - An output layer for regression (no activation function).

        This method should be called before compiling the model.
        """
        self.model = Sequential()
        self.model.add(Input(shape=(1,)))  # Input layer: expects a single feature
        self.model.add(Dense(16, activation='relu'))  # Hidden layer with 16 units
        self.model.add(Dense(32, activation='relu'))  # Hidden layer with 32 units
        self.model.add(Dense(64, activation='relu'))  # Hidden layer with 64 units
        self.model.add(Dense(64, activation='relu'))  # Hidden layer with 64 units
        self.model.add(Dense(128, activation='relu'))  # Hidden layer with 128 units
        self.model.add(Dense(128, activation='relu'))  # Hidden layer with 128 units
        self.model.add(Dense(1))  # Output layer: 1 unit for regression output

    def compile_model(self):
        """
        Compiles the model with the specified loss function, optimizer, and metrics.

        The model is compiled with:
        - Mean squared error loss function.
        - Adam optimizer.
        - Mean absolute error as a metric.

        Raises:
            ValueError: If the model is not built before calling this method.
        
        This method should be called after building the model.
        """
        if self.model is not None:
            self.model.compile(loss="mean_squared_error", optimizer='adam', metrics=['mae'])
        else:
            raise ValueError("Model is not built. Call build_model() first.")