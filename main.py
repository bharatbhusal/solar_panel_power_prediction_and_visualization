from src.data_preprocessor import DataPreprocessor
from src.model_builder import ModelBuilder
from src.model_trainer import ModelTrainer
from src.model_evaluator import ModelEvaluator
from src.predictor import Predictor
from src.visualizer import Visualizer

def main():
    try:
        # Data preprocessing
        preprocessor = DataPreprocessor(data_path='./data/dataset.csv')
        preprocessor.load_data()
        preprocessor.prepare_data()  # Ensure this method prepares data correctly
        preprocessor.split_data()

        # Model building
        model_builder = ModelBuilder()
        model_builder.build_model()
        model_builder.compile_model()  # Ensure model is compiled

        # Model training
        trainer = ModelTrainer(model=model_builder.model, x_train=preprocessor.x_train, y_train=preprocessor.y_train)
        trainer.train_model()

        # Model evaluation
        evaluator = ModelEvaluator(model=model_builder.model, x_test=preprocessor.x_test, y_test=preprocessor.y_test)
        evaluator.evaluate_model()

        # Visualization
        visualizer = Visualizer(data=preprocessor.data)
        visualizer.plot_time_vs_temperature()
        visualizer.plot_temperature_histogram()
        visualizer.plot_time_vs_temperature_line()
        visualizer.plot_temperature_vs_dc_power()
        visualizer.plot_temperature_vs_dc_power_line()
        visualizer.plot_scatter_temperature_vs_dc_power()
        visualizer.plot_dc_power_histogram()

        # Prediction
        predictor = Predictor(model=model_builder.model)
        try:
            temperature = float(input("Please give the Temperature value: "))
            predictor.predict(temperature)
        except ValueError:
            print("Invalid temperature value. Please enter a numeric value.")

    except FileNotFoundError:
        print("Error: The specified dataset file was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()