# Importing packages
from src.logger import logging
from src.exception import CustomException
from src.components.config_entity import DataIngestionConfig
from src.components.config_entity import DataTransformationConfig
from src.components.data_ingestion import DataIngestion


# Running the feature store creating script
if __name__ == '__main__':
    
    # Creating artifacts folder and ingesting the data
    #ingestion_obj = DataIngestion()
    #train_path, test_path = ingestion_obj.initiate_data_ingestion()
    ingest_obj = DataIngestionConfig()
    train_path = ingest_obj.train_data_path
    test_path = ingest_obj.test_data_path