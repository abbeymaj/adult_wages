# Importing packages
import sys
import pytest
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from src.components.config_entity import DataIngestionConfig
from src.utils import remove_blank_spaces
from src.utils import remove_question_mark
from src.utils import recode_target_class
from src.utils import WOE
from src.components.data_transformation import DataTransformation

# Creating a function to define the path of the untransformed train dataset
@pytest.fixture(scope='function')
def train_data_path():
    train_data_config = DataIngestionConfig()
    return train_data_config.train_data_path

# Creating a function to define the path of the untransformed test dataset
@pytest.fixture(scope='function')
def test_data_path():
    test_data_config = DataIngestionConfig()
    return test_data_config.test_data_path

# Creating a function to check whether the object columns do not contain 
# blank spaces in the train dataset
def test_check_blank_spaces_train(train_data_path):
    train_df = pd.read_parquet(train_data_path)
    clean_train_df = remove_blank_spaces(train_df)
    col_list = clean_train_df.select_dtypes(include=['object'])
    for col in col_list:
        assert all(clean_train_df[col].str.contains(r'\s')) is False

# Creating a function to check whether the object columns do not contain
# blank spaces in the test dataset
def test_check_blank_spaces_test(test_data_path):
    test_df = pd.read_parquet(test_data_path)
    clean_test_df = remove_blank_spaces(test_df)
    col_list = clean_test_df.select_dtypes(include=['object'])
    for col in col_list:
        assert all(clean_test_df[col].str.contains(r'\s')) is False

# Creating a function to check whether the question mark ("?") has been 
# removed from the train dataset
def test_check_question_mark_removed_trainset(train_data_path):
    train_df = pd.read_parquet(train_data_path)
    clean_train_df = remove_question_mark(train_df)
    for col in ['workclass', 'occupation', 'native-country']:
        assert '?' not in clean_train_df[col].values

# Creating a function to check whether the question mark ("?") has been
# removed from the test dataset
def test_check_question_marked_removed_testset(test_data_path):
    test_df = pd.read_parquet(test_data_path)
    clean_test_df = remove_question_mark(test_df)
    for col in ['workclass', 'occupation', 'native-country']:
        assert '?' not in clean_test_df[col].values

# Creating a function to recode the target_class column in the train
# dataset and verify that there are two distinct values after recoding.
def test_recode_train_target_class_column(train_data_path):
    train_df = pd.read_parquet(train_data_path)
    train_df = train_df.pipe(remove_blank_spaces).pipe(recode_target_class)
    assert len(list(train_df['target_class'].unique())) == 2

# Creating a function to recode the target_class column in the train
# dataset and verify that the values are numeric
def test_recode_train_target_column_type(train_data_path):
    train_df = pd.read_parquet(train_data_path)
    train_df = train_df.pipe(remove_blank_spaces).pipe(recode_target_class)
    assert pd.api.types.is_numeric_dtype(train_df['target_class']) is True

# Creating a function to recode the target_class in the test
# dataset
def test_recode_test_target_class_column(test_data_path):
    test_df = pd.read_parquet(test_data_path)
    test_df = test_df.pipe(remove_blank_spaces).pipe(recode_target_class)
    assert len(list(test_df['target_class'].unique())) == 2

# Creating a function to recode the target_class column in the test
# dataset and verify that the values are numeric
def test_recode_test_target_column_type(test_data_path):
    test_df = pd.read_parquet(test_data_path)
    test_df = test_df.pipe(remove_blank_spaces).pipe(recode_target_class)
    assert pd.api.types.is_numeric_dtype(test_df['target_class']) is True

# Creating a function to test that the preprocessor object works as expected
def test_preprocessor_object_train_set(train_data_path):
    train_df = pd.read_parquet(train_data_path)
    train_df = train_df.pipe(remove_blank_spaces).pipe(recode_target_class)
    X = train_df.copy().drop(labels=['target_class'], axis=1)
    y = train_df[['target_class']]
    data_transformation = DataTransformation()
    preprocessor_obj = data_transformation.create_data_transformation_object()
    X_fin = preprocessor_obj.fit_transform(X, y)
    assert len(list(X_fin.columns)) == 16