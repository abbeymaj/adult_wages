# Importing packages
import sys
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from src.components.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging


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
            pass
        
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
            pass
        
        except Exception as e:
            raise CustomException(e, sys)