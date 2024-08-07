from src.data_preprocessor import DataPreprocessor
from src.model_builder import ModelBuilder
from src.model_trainer import ModelTrainer

def main():
    model_filepath = './data/saved_model.keras'  # Updated path to use the Keras format

    try:
        # Data preprocessing
        preprocessor = DataPreprocessor(data_path='./data/dataset.csv')
        preprocessor.load_data()
        preprocessor.prepare_data()  # Ensure this method prepares data correctly
        preprocessor.split_data()

        # Model building
        model_builder = ModelBuilder()
        # Build and compile a new model
        model_builder.build_model()
        model_builder.compile_model()  # Ensure model is compiled

        # Model training
        trainer = ModelTrainer(model=model_builder.model, x_train=preprocessor.x_train, y_train=preprocessor.y_train)
        # Train the model and save it, overwriting if it already exists
        trainer.train_model(epochs=5000)
        trainer.save_model(model_filepath)
        print("Trained and saved new model.")

    except FileNotFoundError:
        print("Error: The specified dataset file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()