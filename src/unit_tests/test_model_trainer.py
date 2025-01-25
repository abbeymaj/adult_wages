# Importing packages
import os
import subprocess
import pytest
import pandas as pd
import xgboost as xgb
import mlflow
from src.utils import best_model_callback
from src.components.find_best_model import FindBestModel
from src.components.config_entity import StoreFeatureConfig
from src.components.model_trainer import ModelTrainer


# Creating a fixture to load the train set and target set
@pytest.fixture(scope='function')
def xform_train_data():
    data_path = StoreFeatureConfig()
    dataset = data_path.xform_train_path
    df = pd.read_parquet(dataset)
    df_sample = df.sample(frac=0.25, random_state=42)
    train_set = df_sample.copy().drop(labels=['target_class'], axis=1)
    target_set = df_sample[['target_class']].copy()
    return train_set, target_set

# Creating a function to verify that the study object successfully
# returns a model and a set of hyperparameters
def test_find_best_model(xform_train_data):
    train_set, target_set = xform_train_data
    model = FindBestModel(train_set, target_set, model_callback=best_model_callback)
    best_model, best_params = model.create_study()
    assert best_model is not None
    assert best_params is not None
    assert isinstance(best_model, xgb.core.Booster)
    assert isinstance(best_params, dict)

# Creating a function to verify that the X_train and X_test feature and 
# target sets are created successfully
def test_create_train_test_sets():
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test = trainer.create_feature_target_datasets()
    assert X_train is not None
    assert X_test is not None
    assert y_train is not None
    assert y_test is not None
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.DataFrame)
    assert isinstance(y_test, pd.DataFrame)

# Creating a function to verify that the model training occurs successfully and 
# the best parameters are returned. The model is not saved in this test.
def test_train_model():
    model_trainer = ModelTrainer()
    best_model, best_params, metric, model_path = model_trainer.initiate_model_training(save_model=False)
    assert model_path is not None
    assert best_params is not None
    assert metric is not None
    assert best_model is not None
    assert isinstance(model_path, str)
    assert isinstance(best_params, dict)
    assert isinstance(metric, float)
    assert isinstance(best_model, xgb.core.Booster)

def test_predict_model(xform_train_data):
    subprocess.call('./start_mlflow_server.sh', shell=True)
    version = 3
    model_name = 'training_model'
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{model_name}/{version}"
    )
    y_preds = model.predict(xform_train_data[0])
    assert y_preds is not None
    
    