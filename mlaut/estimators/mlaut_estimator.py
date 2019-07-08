from abc import ABC, abstractmethod
from mlaut.shared.static_variables import (GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                                            GRIDSEARCH_NUM_CV_FOLDS,
                                            VERBOSE)
import pickle
from mlaut.shared.static_variables import PICKLE_EXTENTION
import wrapt
from mlaut.shared.files_io import DiskOperations

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import preprocessing
import logging
import os

class MlautEstimator(ABC):
    """
    Abstact base class that all mlaut estimators should inherit from.
    """


    @property
    def properties(self):
        return self._properties



    def fit(self, metadata, data):
        """
        Calls the estimator fit method

        Parameters
        ----------
        metadata: dictionary
            metadata including the target variable
        data: pandas DataFrame
            training data
        
        Returns
        -------
        sktime estimator:
            fitted estimator
        """
        y = data[metadata['target']]
        X = data.drop(metadata['target'], axis=1)
        return self._estimator.fit(X,y)    


    def predict(self, X):
        """
        Properties
        ----------
        X: dataframe or numpy array
            features on which predictions will be made
        """
        return self._estimator.predict(X)

    def save(self, dataset_name, cv_fold, strategy_save_dir):
        """
        Saves the strategy on the hard drive
        Parameters
        ----------
        dataset_name:string
            Name of the dataset
        cv_fold: int
            Number of cross validation fold on which the strategy was trained
        strategy_save_dir: string
            Path were the strategies will be saved
        """
        if strategy_save_dir is None:
            raise ValueError('Please provide a directory for saving the strategies')
        
        #TODO implement check for overwriting already saved files
        save_path = os.path.join(strategy_save_dir, dataset_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #TODO pickling will not work for all strategies
        pickle.dump(self, open(os.path.join(save_path, self._properties['name'] + '_cv_fold'+str(cv_fold)+ '.p'),"wb"))
    
    def check_strategy_exists(self, 
                              dataset_name, 
                              cv_fold,
                              strategy_save_dir):
        """
        Checks whether a strategy with the same name was already saved on the disk


        Parameters
        ----------
        dataset_name : str
            name of the dataset to check if trained
        cv_fold : int
            cv fold number
        strategy_save_dir : str
            location where the strategies are being saved
        
        Returns
        -------
        bool:
            If true strategy exists
        """
        path_to_check = os.path.join(strategy_save_dir, dataset_name, self._properties['name'] + '_cv_fold'+str(cv_fold)+ '.p')
        return os.path.exists(path_to_check)
    def load(self, path):
        """
        Load saved strategy
        Parameters
        ----------
        path: String
            location on disk where the strategy was saved
        
        Returns
        -------
        strategy:
            sktime strategy
        """
        return pickle.load(open(path,'rb'))
    

    def _create_pipeline(self, estimator):
        """
        Creates a pipeline for transforming the features of the dataset and training the selected estimator.

        Args:
            estimator (sklearn estimator): Reference of sklearn estimator that will be used at the end of the pipeline.


        Returns:
            `estimator(sklearn pipeline or GridSerachCV)`: `sklearn` pipeline object. If no preprocessing was set 
        """

        if 'data_preprocessing' in self.properties:
            data_preprocessing = self.properties['data_preprocessing']

            if data_preprocessing['normalize_labels'] is True:
                pipe = Pipeline(
                    memory=None,
                    steps=[
                        ('standardscaler', preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True) ),
                        ('estimator', estimator)
                        ]
                )
                return pipe
        else:
            return estimator

