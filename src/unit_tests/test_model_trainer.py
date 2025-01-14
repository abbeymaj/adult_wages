# Importing packages
import os
import pytest
import pandas as pd
import xgboost as xgb
from src.utils import best_model_callback
from src.components.find_best_model import FindBestModel
from src.components.config_entity import StoreFeatureConfig

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
    