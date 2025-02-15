# Importing packages
import os
import pytest
import pandas as pd
import mlflow
import dagshub
from src.utils import remove_blank_spaces
from src.components.config_entity import DataIngestionConfig
from src.components.make_prediction import MakePredictions


# Creating a fixture to load the train set and target set
@pytest.fixture(scope='function')
def xform_train_data():
    data_path = DataIngestionConfig()
    dataset = data_path.train_data_path
    df = pd.read_parquet(dataset)
    df_sample = df.sample(frac=0.05, random_state=42)
    train_set = df_sample.copy().drop(labels=['target_class'], axis=1)
    target_set = df_sample[['target_class']].copy()
    return train_set, target_set

# Creating a function to verify that the system can retrieve the model
# parameters.
def test_retrieve_model_params():
    predictor = MakePredictions()
    runs_data, latest_model_uri = predictor.retrieve_model_params()
    assert runs_data is not None
    assert latest_model_uri is not None
    assert isinstance(runs_data, dict)
    assert isinstance(latest_model_uri, str)

# Creating a function to verify that the model can be retrieved from the
# model registry.
def test_retrieve_model():
    predictor = MakePredictions()
    model = predictor.retrieve_model()
    assert model is not None

# Creating a function to verify that the model can make predictions on a
# test set.
def test_make_prediction(xform_train_data):
    train_data, target_data = xform_train_data
    if 'fnlwgt' in list(train_data.columns):
        train_data.drop(labels=['fnlwgt'], axis=1, inplace=True)
    train_data_clean = train_data.pipe(remove_blank_spaces)
    predictor = MakePredictions()
    preds = predictor.predict(train_data_clean)
    assert preds is not None