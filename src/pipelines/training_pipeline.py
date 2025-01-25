# Importing packages
import os
import subprocess
import mlflow
from src.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    subprocess.call('./start_mlflow_server.sh', shell=True)
    with mlflow.start_run(run_name='training_pipeline') as run:
        run_id = run.info.run_id
        trainer = ModelTrainer()
        best_model, best_params, metric, _ = trainer.initiate_model_training(save_model=False)
        mlflow.log_params(best_params)
        mlflow.log_metric('roc_auc_score', metric)
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path='training_model',
            registered_model_name='training_model'
        )
        mlflow.register_model(f"runs:/{run_id}/models/training_model", "training_model")