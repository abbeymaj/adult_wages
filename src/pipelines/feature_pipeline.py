# Importing packages
from src.logger import logging
from src.exception import CustomException
from src.components.config_entity import DataIngestionConfig
from src.components.config_entity import DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.store_features import FeatureStoreCreation


# Running the feature store creating script
if __name__ == '__main__':
    
    # Creating artifacts folder and ingesting the data
    ingestion_obj = DataIngestion()
    train_path, test_path = ingestion_obj.initiate_data_ingestion()
    
    # Transforming the datasets
    data_transf_obj = DataTransformation()
    train_set, test_set = data_transf_obj.initiate_data_transformation(train_path=train_path, test_path=test_path)
    
    # Saving the transformed datasets to the feature store
    feature_store_obj = FeatureStoreCreation()
    feature_store_obj.create_feature_store(train_set=train_set, test_set=test_set)