# Importing packages
import os
import pytest
import pandas as pd
from src.components.config_entity import DataIngestionConfig

# Creating a function to get the path to the train dataset
@pytest.fixture(scope='function')
def train_data_path():
    train_data_config = DataIngestionConfig()
    return train_data_config.train_data_path

# Creating a function to get the path to the test dataset
@pytest.fixture(scope='function')
def test_data_path():
    test_data_config = DataIngestionConfig()
    return test_data_config.test_data_path

# Verifying that the artifacts folder exists
def test_check_artifacts_folder(path='artifacts'):
    assert os.path.exists('artifacts') is True

# Verifying the train dataset file exists
def test_check_train_data_path(train_data_path):
    assert os.path.exists(train_data_path) is True

# Verifying the test dataset file exists
def test_check_test_data_path(test_data_path):
    assert os.path.exists(test_data_path) is True

# Verifying the train dataset has 15 columns
def test_count_train_dataset_columns(train_data_path):
    df_train = pd.read_parquet(train_data_path)
    assert len(list(df_train.columns)) == 15

# Verifying the test dataset has 15 columns
def test_count_test_dataset_columns(test_data_path):
    df_test = pd.read_parquet(test_data_path)
    assert len(list(df_test.columns)) == 15