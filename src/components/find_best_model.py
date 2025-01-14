# Importing packages
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import best_model_callback
import xgboost as xgb
import optuna
from sklearn import set_config
set_config(transform_output='pandas')
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Creating a class to find the best xgboost model
class FindBestModel():
    '''
    This class finds the best xgboost model. The class uses Optuna to
    find the best model. The class has two methods - The first method 
    defines the objective function and the second method uses the objective
    function to find the best model.
    '''
    # Creating the constructor for the class
    def __init__(
        self,
        train_set,
        target_set,
        model_callback=None,
        key='best_booster',
        n_trials=100,
        seed=42
    ):
        '''
        This is the constructor for the class. It sets the train set, target set, 
        model callback (if any), key and number of trials. It also defines the 
        seed.
        '''
        self.train_set = train_set
        self.target_set = target_set
        self.key = key
        self.n_trials = n_trials
        self.seed = seed
        if model_callback is not None:
            self.model_callback = model_callback
    
    # Creating a method to define the objective function for the training
    def objective(self, trial):
        '''
        This method defines the objective function to train the model. The method
        uses Optuna to find the best hyperparameters for the model. The method uses
        the best hyperparameters to find the best model.
        ================================================================================
        -------------------
        Parameters:
        -------------------
        trial : optuna.trial.Trial - This is the trial object that is used by Optuna.
        
        -------------------
        Returns:
        -------------------
        mean_roc_auc_score : float - This is the mean roc auc score for the best model.
        ================================================================================
        '''
        try:
            # Defining the stratified kfold object and creating the matrix for 
            # the train and validation sets
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            auc_scores = []
            for train_idx, val_idx in skf.split(self.train_set, self.target_set):
                X_trn, X_val = self.train_set.iloc[train_idx], self.train_set.iloc[val_idx]
                y_trn, y_val = self.target_set.iloc[train_idx], self.target_set.iloc[val_idx]
                dtrain = xgb.DMatrix(X_trn, label=y_trn)
                dval = xgb.DMatrix(X_val, label=y_val)
                
                # Defining the parameters for the model
                params = {
                    'verbosity': 0,
                    'eval_metric': 'auc',
                    'objective': 'binary:logistic',
                    'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
                    'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
                    'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
                    'seed': self.seed  
                }
                
                if params['booster'] == 'gbtree' or params['booster'] == 'dart':
                    params['max_depth'] = trial.suggest_int('max_depth', 1, 10)
                    params['eta'] = trial.suggest_float('eta', 1e-2, 0.5, log=True)
                    params['gamma'] = trial.suggest_float('gamma', 1e-8, 1.0, log=True)
                    params['grow_policy'] = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
                if params['booster'] == 'dart':
                    params['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
                    params['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
                    params['rate_drop'] = trial.suggest_float('rate_drop', 1e-8, 1.0, log=True)
                    params['skip_drop'] = trial.suggest_float('skip_drop', 1e-8, 1.0, log=True)
                
                # Adding a callback for the pruner
                pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'validation-auc')
                
                # Training the xgboost model
                bst = xgb.train(params, dtrain, evals=[(dval, 'validation')], callbacks=[pruning_callback])
                
                # Predicting on the validation set
                y_pred = bst.predict(dval)
                auc = roc_auc_score(y_val, y_pred)
                auc_scores.append(auc)
                
            # Calculating the mean roc auc score
            mean_auc = np.mean(auc_scores)
            trial.set_user_attr(key=self.key, value=bst)
            return mean_auc
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating a method to locate the best model using the objective function
    def create_study(self):
        '''
        This method uses the study object to find the best model. The method returns
        the best model and best parameters for the best model.
        ================================================================================
        -------------------
        Returns:
        -------------------
        best_model : xgboost.core.Booster - This is the best model for the dataset.
        best_params : dict - This is the best parameters for the best model.
        ================================================================================
        '''
        try:
            study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction='maximize')
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=[self.model_callback])
            best_model = study.user_attrs[self.key]
            best_params = study.best_params
            return best_model, best_params
        
        except Exception as e:
            raise CustomException(e, sys)

