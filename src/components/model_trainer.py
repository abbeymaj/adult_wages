# Importing packages
import sys
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from src.utils import best_model_callback
from src.utils import save_object
from src.utils import make_predictions
from src.components.config_entity import DataTransformationConfig
from src.components.config_entity import StoreFeatureConfig
from src.components.config_entity import ModelTrainerConfig
from src.components.find_best_model import FindBestModel
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import roc_auc_score


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
        # Instantiating the path where the trained model will be saved
        self.trained_model_path = ModelTrainerConfig()
        # Instantiating the path to the preprocessor object
        self.preprocessor_path = DataTransformationConfig()
        # Instantiating the path to the feature store
        self.data_path = StoreFeatureConfig()
    
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
            # Reading the train and test datasets
            train_data_set = pd.read_parquet(self.data_path.xform_train_path)
            test_data_set = pd.read_parquet(self.data_path.xform_test_path)
            
            # Creating the train feature and target sets
            X_train = train_data_set.copy().drop(columns=['target_class'], axis=1)
            y_train = train_data_set[['target_class']]
            
            # Creating the test feature and target sets
            X_test = test_data_set.copy().drop(columns=['target_class'], axis=1)
            y_test = test_data_set[['target_class']]
            
            return (
                X_train,
                X_test,
                y_train,
                y_test
            )
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating the method to initiate the model training process
    def initiate_model_training(self, save_model=True):
        '''
        This method trains the model and then saves the trained model to the artifacts
        folder.
        ===================================================================================
        ------------------------
        Parameters:
        ------------------------
        save_model : bool - This determines if the model should be saved or not.
        
        ------------------------
        Returns:
        ------------------------
        model_path : str - This is the path to the saved model.
        best_params : dict - This is the best hyperparameters for the best model.
        metric : float - This is the metric from the prediction.
        ====================================================================================
        '''
        try:
            # Fetching the datasets
            X_train, X_test, y_train, y_test = self.create_feature_target_datasets()
            
            # Instantiating the FindBestModel class
            model = FindBestModel(
                train_set=X_train,
                target_set=y_train,
                model_callback=best_model_callback
            )
            
            # Getting the best model and the best hyperparameters
            best_model, best_params = model.create_study()
            
            # Saving the best model
            if save_model is True:
                save_object(
                    file_path=self.trained_model_path.model_path,
                    object=best_model
                )
            
            # Making predictions using the test set
            y_preds = make_predictions(
                dataset=X_test,
                model=best_model
            )
            
            # Calculating the roc auc score
            metric = roc_auc_score(y_test, y_preds)
            
            return (
                best_model,
                best_params,
                metric,
                self.trained_model_path.model_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)