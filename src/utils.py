# Importing packages
import os
import pathlib
import sys
import dill
import datetime
import json
import mlflow
from src.exception import CustomException
from category_encoders import WOEEncoder
import sklearn
sklearn.set_config(transform_output='pandas')
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import xgboost as xgb


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
def recode_target_class(df):
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
    df.loc[:, 'target_class'] = df.loc[:, 'target_class'].map(lambda x: 0 if x == '<=50K' else 1)
    df['target_class'] = df.loc[:, 'target_class'].astype(int)
    return df
    

# Creating a class to encode variables based on Weight of Evidence
class WOE(TransformerMixin, ClassifierMixin, BaseEstimator):
    '''
    This class encodes categorical variables using the weight of evidence. This class has
    two methods - a fit method and a transform method. The class also inherits from 
    sklearn's BaseEstimator and TransformerMixin classes.
    '''
    def __init__(self, cols=None):
        '''
        This is the constructor of the weight of evidence class. It instantiates the columns
        which will be transformed using the weight of evidence.  
        '''
        self.cols = cols
    
    def fit(self, X, y):
        '''
        This method uses the feature and the target set to fit the identified categorical 
        columns with the calculated weight of evidence per categorical variable.
        ========================================================================================
        ---------------------
        Parameters:
        ---------------------
        X : This is the feature dataset containing the categorical variables.
        y : This is the target dataset.
        
        ---------------------
        Returns:
        ---------------------
        The dataset after being fit with the data.
        =========================================================================================
        
        '''
        self.woe_encoder = WOEEncoder(cols=self.cols)
        self.woe_encoder.fit(X, y)
        return self
        
    
    def transform(self, X, y=None):
        '''
        This method transforms the categorical data into their calculated weight of 
        evidence after the data has been fit. If the target dataset is present,
        the method executes the transformation using both the feature and target
        datasets. Else, the method will only use the feature dataset to transform the
        dataset.
        ========================================================================================
        ---------------------
        Parameters:
        ---------------------
        X : This is the feature dataset containing the categorical variables.
        y : This is the target dataset. This is not a mandatory arguement. 
        
        ---------------------
        Returns:
        ---------------------
        The dataset after being transformed using the weight of evidence.
        =========================================================================================
        '''
        if y is not None:
            return self.woe_encoder.transform(X, y)
        else:
            return self.woe_encoder.transform(X)
    
    @classmethod
    def __sklearn_tags__(cls):
        return {
            'non_deterministic': False,
            'requires_positive_X': False,
            'requires_positive_y': False,
            'X_types': ['2darray', 'string'],
            'poor_score': False,
            'no_validation': False,
            'multioutput': False,
            'allow_nan': False,
            'stateless': False,
            'multilabel': False,
            '_skip_test': False,
            'multioutput_only': False,
            'binary_only': False,
            'requires_fit': True,
        }


# Creating a function to convert capital gains and capital loss into a categorical feature
def convert_to_categorical(df):
    '''
    This function will convert the capital-gains and capital-loss features from a 
    continuous feature to a categorical feature.
    ========================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : This is the dataset with the capital-gain and capital-loss features
    
    ---------------------
    Returns:
    ---------------------
    df : The modified dataset after transforming the capital-gain and capital-loss features
    into a categorical feature.
    =========================================================================================
    '''
    # Transforming the capital-gain feature into a categorical feature
    df.loc[:, 'capital-gain-trns'] = df.loc[:, 'capital-gain'].map(lambda a: 'cap_gain' if a > 0 else 'no_cap_gain')
    df.drop(labels=['capital-gain'], axis=1, inplace=True)
    
    # Transforming the capital-loss feature into a categorical feature
    df.loc[:, 'capital-loss-trns'] = df.loc[:, 'capital-loss'].map(lambda b: 'cap_loss' if b > 0 else 'no_cap_loss')
    df.drop(labels=['capital-loss'], axis=1, inplace=True)
    return df

# Creating a custom callback to use when finding the best xgboost model
def best_model_callback(study, trial):
    '''
    This function creates a custom callback to use when searching for the 
    best XGBoost model. The function is used in conjuction with Optuna to 
    find the best model.
    ========================================================================================
    ---------------------
    Parameters:
    ---------------------
    study : This is the study object from Optuna.
    trial : This is the number of trials when optimizing the Optuna study.
    =========================================================================================
    '''
    if study.best_trial.number == trial.number:
        study.set_user_attr(key='best_booster', value=trial.user_attrs['best_booster'])


# Creating a function to make predictions using the best model
def make_predictions(dataset, model):
    '''
    This function makes predictions, given a dataset and a model. The function coverts a 
    dataset into a DMatrix and then makes predictions using the model.
    ========================================================================================
    ---------------------
    Parameters:
    ---------------------
    dataset : This is the dataset on which predictions need to be made.
    model : This is the model that will be used to make the predictions.
    
    ---------------------
    Returns:
    ---------------------
    y_pred : This is the prediction made by the model.
    =========================================================================================
    '''
    dmatrix = xgb.DMatrix(dataset)
    y_pred = model.predict(dmatrix)
    return y_pred


# Creating a function to set the tracking uri
def set_tracking_uri():
    '''
    This function sets the tracking uri for the MLflow server.
    ========================================================================================
    ---------------------
    Returns:
    ---------------------
    Sets the tracking uri for the Mlflow server.
    =========================================================================================
    '''
    model_uri = pathlib.Path().cwd().parent / 'model_db' / 'mlflow.db'
    return mlflow.set_tracking_uri(model_uri)


# Creating a function to save the run parameters as a json file
def save_run_params(run_params):
    '''
    This function saves the run parameters as a json file in the run_config folder. 
    ========================================================================================
    ---------------------
    Parameters:
    ---------------------
    run_params : dict - This is the dictionary containing the run parameters.
    
    ---------------------
    Returns:
    ---------------------
    Saves the run parameters as a json file into the run_config folder
    '''
    now = datetime.datetime.now().strftime('%Y%m%d')
    file_path = pathlib.Path().cwd() / 'run_config' / f'run_params_{now}.json'
    with open(file_path, 'w') as file_obj:
        json.dump(run_params, file_obj)
        

# Creating a function to load the run parameters json file.
def load_run_params(directory='run_config'):
    '''
    This function loads the run parameters as a json file, which is present
    in the run_config folder. 
    ========================================================================================
    ---------------------
    Parameters:
    ---------------------
    directory : str - This is the name of the directory in which the run parameters json
    file is stored.
    
    ---------------------
    Returns:
    ---------------------
    run_parameters : json - This is the run parameters json file. 
    '''
    dir_path = pathlib.Path().cwd() / directory
    json_files = os.listdir(dir_path)
    latest_file = None
    latest_date = None
    for file_name in json_files:
        date_str = file_name.split('_')[2].split('.')[0]
        file_date = datetime.datetime.strptime(date_str, '%Y%m%d')
        if not latest_date or file_date > latest_date:
            latest_date = file_date
            latest_file = dir_path / file_name
    return latest_file


# Creating a function to read the JSON file
def read_json_file(file_path):
    '''
    This function reads the run parameters JSON file and returns the contents of the file.
    ========================================================================================
    ---------------------
    Parameters:
    ---------------------
    file_path : str - This is the path to the run parameters json file.
    
    ---------------------
    Returns:
    ---------------------
    run_parameters : json - This is the run pararmeters json file.
    '''
    with open(file_path, 'r') as file_obj:
        data = json.load(file_obj)
    return data