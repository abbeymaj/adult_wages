# Importing packages
import sys
import pandas as pd
from src.exception import CustomException

# Creating a class to convert the user entered data into a pandas dataframe
class CustomData():
    '''
    This class is responsible for converting the user entered data into a pandas
    dataframe. This will assist in transforming the data and using the transformed
    data to make predictions.
    '''
    # Creating the constructor for the class
    def __init__(
        self,
        age:int,
        workclass:str,
        education:str,
        education_num:int,
        marital_status:str,
        occupation:str,
        relationship:str,
        race:str,
        sex:str,
        capital_gain:int,
        capital_loss:int,
        hours_per_week:int,
        native_country:str
    ):
        '''
        This is the constructor for the custom data class.
        '''
        self.age = age
        self.workclass = workclass
        self.education = education
        self.education_num = education_num
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.capital_gain = capital_gain
        self.capital_loss = capital_loss
        self.hours_per_week = hours_per_week
        self.native_country = native_country
    
    # Creating a method to convert the user entered data into a dataframe.
    def create_dataframe(self):
        '''
        This method takes the data input by the user and returns a dataframe. The method 
        converts the data, input by the user on the website, into a dictionary and then creates
        a pandas dataframe using the dictionary.
        ========================================================================================
        -----------------------
        Returns:
        -----------------------
        df : pandas dataframe - A pandas dataframe of the data entered by the user.
        ========================================================================================
        '''
        try:
            # Converting the data into a dictionary
            custom_data_input_dict = {
                'age': [self.age],
                'workclass': [self.workclass],
                'education': [self.education],
                'education-num': [self.education_num],
                'marital-status': [self.marital_status],
                'occupation': [self.occupation],
                'relationship': [self.relationship],
                'race': [self.race],
                'sex': [self.sex],
                'capital-gain': [self.capital_gain],
                'capital-loss': [self.capital_loss],
                'hours-per-week': [self.hours_per_week],
                'native-country': [self.native_country]
            }
            
            # Creating a dataframe from the dictionary
            df = pd.DataFrame(custom_data_input_dict)
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)