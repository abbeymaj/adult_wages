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
        This function will ingest the data from source, split the dataset into a train
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
        pass
