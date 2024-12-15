# Importing packages
import sys
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from src.components.config_entity import DataIngestionConfig
from src.components.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils import WOE
from src.utils import convert_to_categorical
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# Creating a class to transform the train and test datasets
class DataTransformation():
    '''
    This class contains methods to preprocess the train an test datasets and
    save the preprocessing object. 
    '''
    # Creating the constructor for the data transformation class
    def __init__(self):
        '''
        This constructor instantiates the path into which the preprocessor
        object will be stored.
        '''
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
    
    # Creating a method to create the preprocessing object.
    def create_data_transformation_object(self):
        '''
        This method creates the preprocessor object.
        =============================================================================
        ------------------
        Returns:
        ------------------
        preprocessor : pipeline object - Returns the preprocessor object. 
        =============================================================================
        '''
        try:
            # Reading the path for the train dataset
            #train_data = pd.read_parquet(self.data_ingestion_config.train_data_path)
            
            # Defining the numberical features
            num_cols = ['age', 'education-num', 'hours-per-week']
            
            # Defining the sex feature for one hot encoding
            col_sex_ohe = ['sex']
            
            # Defining the capital gain and loss features for categorization and one hot encoding 
            ohe_cap_gain_loss_cols = ['capital-gain', 'capital-loss']
            
            # Defining the features to use weight of evidence encoding
            woe_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']
            
            # Creating the pipeline for the numerical features
            num_pipeline = Pipeline(
                steps=[
                    ('std', StandardScaler())
                ]
            )
            
            # Creating a pipeline for the sex feature
            ohe_sex_pipeline = Pipeline(
                steps=[
                    ('ohe_sex', OneHotEncoder(sparse_output=False))
                ]
            )
            
            # Creating a pipeline for the capital-gain and capital-loss features
            ohe_cap_pipeline = Pipeline(
                steps=[
                    ('cap_ft', FunctionTransformer(convert_to_categorical)),
                    ('cap_ohe', OneHotEncoder(sparse_output=False))
                ]
            )
            
            # Creating a pipeline for the weight of evidence encoding
            woe_pipeline = Pipeline(
                steps=[
                    ('woe', WOE())
                ]
            )
            
            # Combining all pipelines into a single preprocessor object
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('ohe_sex_pipeline', ohe_sex_pipeline, col_sex_ohe),
                    ('ohe_cap_pipeline', ohe_cap_pipeline, ohe_cap_gain_loss_cols),
                    ('woe_pipeline', woe_pipeline, woe_cols)
                ]
            )
            
            return preprocessor
            
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating a method to initiate the data transformation
    def initiate_data_transformation(self, train_path:str, test_path:str):
        '''
        This method performs the data transformation on the feature set.
        ===============================================================================
        ----------------
        Parameters:
        ----------------
        train_path : str - The path in which the training data is stored.
        test_path : str - The path in which the test data is stored.
        
        ----------------
        Returns:
        ----------------
        train_set : parquet file - The file used for training the model.
        test_set : parquet file - The array used for testing the model.
        preprocessor object path : str - The path in which the preprocessor object 
        is stored.
        ================================================================================
        '''
        try:
            # Reading the train and test datasets
                        
            # Dropping fnlwgt features if the features 
            # exist in the dataset
            pass
        
        except Exception as e:
            raise CustomException(e, sys)