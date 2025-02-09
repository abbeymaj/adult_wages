# Importing packages
import os
import pathlib
import subprocess
import pytest
import pandas as pd
import xgboost as xgb
import mlflow
import dagshub
from src.utils import best_model_callback
from src.components.find_best_model import FindBestModel
from src.components.config_entity import StoreFeatureConfig
from src.components.model_trainer import ModelTrainer
from src.utils import load_run_params
from src.utils import read_json_file


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

# Creating a function to verify if the run_config directory is created and exists
def test_is_run_config_dir_created():
    run_config_dir_path = pathlib.Path().cwd() / 'run_config'
    assert os.path.exists(run_config_dir_path)

# Creating a function to verify if the model parameters can be accessed and downloaded
def test_get_run_parameter_json():
    run_params_json = load_run_params()
    assert run_params_json is not None
    

# Creating a function to verify if the JSON file can be read
def test_read_json_file():
    run_params_json = load_run_params()
    json_data = read_json_file(run_params_json)
    assert json_data is not None

# Creating a function to verify that the model is accessible and can be used for
# inference and predictions.
def test_predict_model(xform_train_data):
    train_set, target_set = xform_train_data
    run_params_json = load_run_params()
    run_data = read_json_file(run_params_json)
    #print(run_data)
    #subprocess.call('./src/unit_tests/start_mlflow_server.sh', shell=True)
    #model_uri = pathlib.Path().cwd() / 'model_db' / 'mlflow.db'
    #model_uri = pathlib.Path('model_db/mlflow.db').resolve()
    dagshub.init(repo_owner='abbeymaj', repo_name='my-first-repo', mlflow=True)
    model_uri = 'https://dagshub.com/abbeymaj/my-first-repo.mlflow'
    mlflow.set_tracking_uri(model_uri)
    my_model_uri = run_data['model_uri']
    model_name = run_data['model_name']
    model_version = run_data['model_version']
    #mlflow.set_experiment('training_2')
    model = mlflow.pyfunc.load_model(my_model_uri)
    assert model is not None
    #y_preds = model.predict(train_set)
    #assert y_preds is not None