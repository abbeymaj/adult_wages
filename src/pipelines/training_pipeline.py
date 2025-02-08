# Importing packages
import pathlib
import subprocess
import dagshub
import mlflow
from mlflow import MlflowClient
from src.utils import save_run_params
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    
    # Initiating the Dagshub client
    dagshub.init(repo_owner='abbeymaj', repo_name='my-first-repo', mlflow=True)
    
    # Setting the tracking uri for the model
    model_uri = 'https://dagshub.com/abbeymaj/my-first-repo.mlflow'
    mlflow.set_tracking_uri(model_uri)
    
    # Instantiating the mlflow client
    client = MlflowClient()
    
    # Creating the experiment
    experiment_id = client.create_experiment('training_1')
        
    # Starting the training run
    run_params = {}
    with mlflow.start_run(run_name='training_pipeline_1', experiment_id=experiment_id) as run:
        # Fetching the run id
        run_id = run.info.run_id
        # Instantiating the model trainer
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
        
        # Fetch the latest version of the model and the model name
        latest_version_info = client.get_latest_versions('training_model_1', stages=['None'])[0]
        model_name = latest_version_info.name
        latest_version = latest_version_info.version
        
        # Storing the model uri and run id into a dictionary
        run_params['model_uri'] = model_info.model_uri
        run_params['run_id'] = run_id
        run_params['model_name'] = model_name
        run_params['model_version'] = latest_version
    
    # Saving the run parameters into a JSON file for future retrieval
    save_run_params(run_params)