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
from src.utils import remove_blank_spaces
from src.utils import recode_target_class
from src.utils import save_object
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
            logging.info('Creating the preporcessor object.')
                 
            # Defining the numerical features
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
            
            logging.info('Preprocessor object has been created successfully.')
            
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
            logging.info('Initiating the data transformation process.')
            
            # Reading the train and test datasets
            train_df = pd.read_parquet(self.data_ingestion_config.train_data_path)
            test_df = pd.read_parquet(self.data_ingestion_config.test_data_path)
                        
            # Instantiating the preprocessor object
            preprocessor_obj = self.create_data_transformation_object()
            
            # Dropping fnlwgt feature if this feature exists in the train dataset
            if 'fnlwgt' in list(train_df.columns):
                train_df.drop(labels=['fnlwgt'], axis=1, inplace=True)
            
            # Dropping the fnlwgt feature if this feature exists in the test dataset
            if 'fnlwgt' in list(test_df.columns):
                test_df.drop(labels=['fnlwgt'], axis=1, inplace=True)
            
            # Removing any spaces in the features and recoding the target class in the 
            # train dataset
            train_df_clean = train_df.pipe(remove_blank_spaces).pipe(recode_target_class)
            
            # Removing any spaces in the features and recoding the target class in the 
            # test dataset
            test_df_clean = test_df.pipe(remove_blank_spaces).pipe(recode_target_class)
            
            # Creating train feature and target sets
            input_feature_train_df = train_df_clean.drop(labels=['target_class'], axis=1)
            input_target_train_df = train_df_clean[['target_class']]
            
            # Creating test feature and target sets
            input_feature_test_df = test_df_clean.drop(labels=['target_class'], axis=1)
            input_target_test_df = test_df_clean[['target_class']]
            
            # Transforming the train and test datasets
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df, input_target_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # Combining the transformed train and test datasets with their respective
            # target sets
            input_train_combined = pd.concat([input_feature_train_arr, input_target_train_df], axis=1)
            input_test_combined = pd.concat([input_feature_test_arr, input_target_test_df], axis=1)
            
            # Saving the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_path,
                object=preprocessor_obj
            )
            
            logging.info('Data transformation process has been completed.')
            
            return(
                input_train_combined,
                input_test_combined
            )
        
        except Exception as e:
            raise CustomException(e, sys)