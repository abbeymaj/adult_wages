# Importing packages
import os
import sys
import pandas as pd
from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException
from src.components.config_entity import DataIngestionConfig

# Creating a class to ingest the data from source
class DataIngestion():
    '''
    This class reads the data from source and splits the data into a train and test set.
    The class has two methods - The constructor, which defines the path in which the 
    datasets will be saved, and the method to ingest the data.
    '''
    # Defining the constructor for the class
    def __init__(self):
        '''
        This is the constructor for the data ingestion class. It defines the path in which
        the train and test dataset will be saved.
        '''
        self.ingestion_config = DataIngestionConfig()
    
    # Defining the function to ingest the data
    def initiate_data_ingestion(self):
        '''
        This method will ingest the data from source, split the dataset into a train
        and test dataset. The function will also create the artifacts folder and store
        the train and test dataset in the artifacts folder.
        ====================================================================================
        ---------------
        Returns:
        ---------------
        train file path : str - This is the path to the train dataset.
        test file path : str - This is the path to the test dataset.
        ====================================================================================
        '''
        logging.info("Beginning the data ingestion process.")
        
        try:
            # Creating the artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Defining the url where the data is located
            URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            
            # Defining the name of the columns for the dataframe
            COLS = [
                'age', 
                'workclass', 
                'fnlwgt', 
                'education', 
                'education-num', 
                'marital-status', 
                'occupation', 
                'relationship', 
                'race', 
                'sex', 
                'capital-gain', 
                'capital-loss', 
                'hours-per-week', 
                'native-country', 
                'target_class'
                ]
            
            # Reading the dataset into a pandas dataframe
            df = pd.read_csv(URL, header=None, names=COLS)
            
            # Splitting the dataset into a train and test set
            train_set, test_set = train_test_split(df, test_size=0.33, random_state=42)
            
            # Saving the datasets into the artifacts folders
            train_set.to_parquet(self.ingestion_config.train_data_path, index=False, compression='gzip')
            test_set.to_parquet(self.ingestion_config.test_data_path, index=False, compression='gzip')
            
            logging.info("Completed the data ingestion process.")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
                
        except Exception as e:
            raise CustomException(e, sys)
