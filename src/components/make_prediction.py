# Importing packages
import sys
import pandas as pd
import mlflow
import dagshub
from src.utils import load_object
from src.utils import load_run_params
from src.utils import read_json_file
from src.components.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging

# Creating a class to make predictions on data received from the website
class MakePredictions():
    '''
    This class is responsible for making predictions on the data received from 
    the website.
    '''
    # Creating the constructor for the class
    def __init__(self):
        '''
        This is the constructor for the MakePredictions class.
        '''
        self.preprocessor_obj = DataTransformationConfig()
        self.model_uri = 'https://dagshub.com/abbeymaj/my-first-repo.mlflow'
    
    # Retrieve the latest parameters for the the trained model
    def retrieve_model_params(self):
        '''
        This method retrieves the model parameters for the latest trained model.
        ===================================================================================
        ----------------
        Returns:
        ----------------
        runs_data : json - This is the json file containing the model parameters for the 
        latest model.
        
        latest_model_uri : str - This is the uri for the latest model.
        ===================================================================================
        '''
        try:
            # Fetching the json file for the latest model
            model_params_json = load_run_params()
            
            # Reading the json file
            runs_data = read_json_file(model_params_json)
            
            # Fetch the model uri for the latest model
            latest_model_uri = runs_data['model_uri']
            
            return runs_data, latest_model_uri

        except Exception as e:
            raise CustomException(e, sys)
    
    
    # Creating a method to retrieve the model from the model registry
    def retrieve_model(self):
        '''
        This method retrieves the trained model from the model registry.
        ===================================================================================
        ----------------
        Returns:
        ----------------
        model : xgboost.core.Booster - This is the trained model from the model registry.
        ===================================================================================
        '''
        try:
            # Initializing the dagshub connection to the model registry
            dagshub.init(repo_owner='abbeymaj', repo_name='my-first-repo', mlflow=True)
            
            # Setting the tracking uri to the trained model
            mlflow.set_tracking_uri(self.model_uri)
            
            # Retrieving the latest model uri
            _, latest_model_uri = self.retrieve_model_params()
            
            # Fetching the model from the model registry
            model = mlflow.pyfunc.load_model(latest_model_uri)
            
            return model
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    # Creating  a method to make predictions on the received data
    def predict(self, features):
        '''
        This method makes predictions using the feature inputs from the web page and 
        the trained model. This method also transforms the input data using the 
        preprocessor object before making the predictions.
        ============================================================================================
        -------------------
        Parameters:
        -------------------
        features : pandas dataframe - This is the feature data input received from the web page.
        
        -------------------
        Returns:
        -------------------
        preds : This is the prediction based on the input features.
        =============================================================================================
        '''
        try:
            # Instantiating the preprocessor object
            preprocessor = load_object(self.preprocessor_obj.preprocessor_obj_path)
            
            # Retrieving the model from the model registry
            model = self.retrieve_model()
            
            # Transforming the features using the preprocessor object
            xform_features = preprocessor.transform(features)
            
            # Making predictions using the transformed features
            preds = model.predict(xform_features)
            
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)
