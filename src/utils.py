# Importing packages
import os
import sys
import dill
from src.exception import CustomException


# Creating a function to save objects as pickle files
def save_object(file_path:str, object):
    '''
    This function saves as object to the given file path.
    ================================================================================
    --------------------
    Parameters:
    --------------------
    file_path : str - This is path to the folder in which the object will be saved.
    object - This is the object, which will be saved.
    
    --------------------
    Returns:
    --------------------
    Saves the object into folder given in the file path. 
    =================================================================================
    '''
    try:
        # Checking if the directory exist, and if not, create a new directory
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Saving the object into the given file path
        with open(file_path, 'wb') as file_obj:
            dill.dump(object, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


# Creating a function to load a saved object 
def load_object(file_path:str):
    '''
    This function will load a pickle object.
    ===================================================================================
    ---------------------
    Parameters:
    ---------------------
    file_path : str - This is the path where the object is stored.
    
    ---------------------
    Returns:
    ---------------------
    The function returns the object after it is loaded.
    ==================================================================================== 
    '''
    try:
        # Reading the file path and loading the pickled object
        with open(file_path, 'rb') as file_obj:
            return dill.load(file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)


# Creating a function to remove blank spaces in front of the text fields
def remove_blank_spaces(df):
    '''
    This function removes any blank spaces, which are present in the text 
    fields of the dataframe.
    ======================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pandas dataframe - This is the original dataframe, for which the blank spaces 
    needs to be removed.
    
    ---------------------
    Returns:
    ---------------------
    df : pandas dataframe - This is the transformed pandas dataframe.
    =======================================================================================
    '''
    df_list = list(df.select_dtypes(include=['object']))
    for col in df_list:
        df[col] = df[col].str.strip()
    return df


# Creating the function to remove "?" from certain columns in the dataset
def remove_question_mark(df):
    '''
    This function removes any question marks ("?") from certain columns in the dataset.
    The question mark will be replaced with the highest frequency in the respective
    column.
    ======================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pandas dataframe - This is the original dataframe, for which the question mark 
    needs to be removed.
    
    ---------------------
    Returns:
    ---------------------
    df : pandas dataframe - This is the transformed pandas dataframe.
    =======================================================================================
    '''
    df.loc[df.loc[:, 'workclass']=='?', 'workclass'] = 'Private'
    df.loc[df.loc[:, 'occupation']=='?', 'occupation'] = 'Prof-specialty'
    df.loc[df.loc[:, 'native-country']=='?', 'native-country'] = 'United-States'
    return df


# Creating a function to recode the target class
def recode_target_class(target):
    '''
    This function recodes the target column from a string column to a numeric binary
    column so that machine learning models can consume it.
    ======================================================================================
    ---------------------
    Parameters:
    ---------------------
    target : This is the target_class column with string values.
    
    ---------------------
    Returns:
    ---------------------
    The target_class column recoded into a binary column.
    =======================================================================================
    '''
    if target == '<=50K':
        return 0
    else:
        return 1