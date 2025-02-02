# Importing packages
import pathlib
import subprocess
import mlflow
from mlflow import MlflowClient
from src.utils import save_run_params
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    # Starting the mlflow server
    subprocess.call('./start_mlflow_server.sh', shell=True)
    
    # Setting the tracking uri for the model
    model_uri = pathlib.Path().cwd() / 'model_db' / 'mlflow.db'
    mlflow.set_tracking_uri(model_uri)
    
    # Instantiating the mlflow client
    client = MlflowClient()
    
    # Creating the experiment
    client.create_experiment('training_1')
        
    # Starting the training run
    run_params = {}
    with mlflow.start_run(run_name='training_pipeline') as run:
        # Fetching the run id
        run_id = run.info.run_id
        # Instantiating the model traniner
        trainer = ModelTrainer()
        # Fetching the best model and best model parameters
        best_model, best_params, metric, _ = trainer.initiate_model_training(save_model=False)
        # Logging the best model, best metrics and best model parameters into Mlflow DB
        mlflow.log_params(best_params)
        mlflow.log_metric('roc_auc_score', metric)
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path='models/training_model_1',
            registered_model_name='training_model_1'
        )
        # Registering the model
        mlflow.register_model(f"runs:/{run_id}/models/training_model_1", "training_model_1")
        # Storing the model uri and run id into a dictionary
        run_params['model_uri'] = model_info.model_uri
        run_params['run_id'] = run_id
    
    # Saving the run parameters into a JSON file for future retrieval
    save_run_params(run_params)