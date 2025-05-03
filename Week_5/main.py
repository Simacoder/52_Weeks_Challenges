import logging
import yaml
import mlflow
import mlflow.sklearn
import pandas as pd
from steps.ingest import Ingestion
from steps.clean import Cleaner
from steps.train import Trainer
from steps.predict import Predictor
from sklearn.metrics import classification_report

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def main():
    try:
        # Load configuration file
        with open('config.yml', 'r') as file:
            config = yaml.safe_load(file)

        mlflow.set_experiment("Model Training Experiment")

        with mlflow.start_run() as run:
            # Step 1: Load data
            ingestion = Ingestion()
            train, test = ingestion.load_data()
            logging.info("✅ Data ingestion completed successfully")

            # Step 2: Clean data
            cleaner = Cleaner()
            train_data = cleaner.clean_data(train)
            test_data = cleaner.clean_data(test)
            logging.info("✅ Data cleaning completed successfully")

            # Step 3: Prepare and train model
            trainer = Trainer()
            X_train, y_train = trainer.feature_target_separator(train_data)

            # Debugging: Check training data
            logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            logging.info(f"First few y_train values: {y_train[:5]}")

            # Ensure y_train is 1D
            y_train = y_train.dropna().ravel()

            # Convert categorical features if needed
            if isinstance(X_train, pd.DataFrame):
                X_train = pd.get_dummies(X_train)

            # Ensure X_test has the same columns as X_train
            X_test, y_test = Predictor().feature_target_separator(test_data)
            y_test = y_test.dropna().ravel()
            if isinstance(X_test, pd.DataFrame):
                X_test = pd.get_dummies(X_test)

            # Align columns to avoid missing feature errors
            X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

            # Debugging: Check final column names
            logging.info(f"Final X_train columns: {X_train.columns.tolist()}")
            logging.info(f"Final X_test columns: {X_test.columns.tolist()}")

            # Train the model
            trainer.train_model(X_train, y_train)
            trainer.save_model()
            logging.info("✅ Model training completed successfully")

            # Step 4: Evaluate model
            predictor = Predictor()
            accuracy, class_report, roc_auc_score = predictor.evaluate_model(X_test, y_test)
            report = classification_report(y_test, trainer.pipeline.predict(X_test), output_dict=True)
            logging.info("✅ Model evaluation completed successfully")

            # Step 5: MLflow logging
            mlflow.set_tag('Model developer', 'prsdm')
            mlflow.set_tag('preprocessing', 'OneHotEncoder, Standard Scaler, and MinMax Scaler')

            # Log model parameters and metrics
            model_params = config.get('model', {}).get('params', {})
            mlflow.log_params(model_params)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("roc", roc_auc_score)
            mlflow.log_metric('precision', report['weighted avg']['precision'])
            mlflow.log_metric('recall', report['weighted avg']['recall'])
            mlflow.sklearn.log_model(trainer.pipeline, "model")

            # Register the model in MLflow
            model_name = "insurance_model"
            model_uri = f"runs:/{run.info.run_id}/model"
            mlflow.register_model(model_uri, model_name)

            logging.info("✅ MLflow tracking completed successfully")

            # Print evaluation results
            print("\n============= Model Evaluation Results ==============")
            print(f"Model: {trainer.model_name}")
            print(f"Accuracy Score: {accuracy:.4f}, ROC AUC Score: {roc_auc_score:.4f}")
            print(f"\n{class_report}")
            print("=====================================================\n")

    except Exception as e:
        logging.error(f"❌ Error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()
