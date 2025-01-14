# Importing packages
import sys
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from src.components.config_entity import DataTransformationConfig
from src.components.config_entity import StoreFeatureConfig
from src.components.config_entity import ModelTrainerConfig
from src.exception import CustomException
from src.logger import logging


# Creating a class to train the model
class ModelTrainer():
    '''
    This class contains methods to train the model and then save the trained model
    in the artifacts folder. 
    '''
    # Creating the constructor for the class
    def __init__(self):
        '''
        This is the constructor of the class. The constructor instantiates the path to 
        the transformed data and the preprocessor object.
        '''
        self.model_trainer_config = ModelTrainerConfig()
    
    # Creating a method to define the feature and target datasets
    def create_feature_target_datasets(self):
        '''
        This method creates the feature and target datasets.
        ============================================================================       
        -------------------
        Returns:
        -------------------
        X_train : pandas dataframe - The training feature set.
        y_train : pandas dataframe - The training target set.
        X_test : pandas dataframe - The test feature set.
        y_test : pandas dataframe - The test target set.
        =============================================================================
        '''
        try:
            pass
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating the method to initiate the model training process
    def initiate_model_training(self):
        '''
        This method trains the model and then saves the trained model to the artifacts
        folder.
        ===================================================================================
        ------------------------
        Returns:
        ------------------------
        model_path : str - This is the path to the saved model.
        metric : float - This is the metric from the prediction.
        ====================================================================================
        '''
        try:
            pass
        
        except Exception as e:
            raise CustomException(e, sys)