import numpy as np
from abc import ABC, abstractmethod

class MLaut_resampling:
    """
    Abstact class that all MLaut resampling strategies should inherint from
    """
    @abstractmethod
    def split(self):
        """
        
        """
class Single_Split(MLaut_resampling):

    def __init__(self, cv):
        """
        Parameters
        ----------
        cv: sktime.model_selection object
            split object

        """
        self._cv = cv
    def split(self, X):
        """
        Parameters
        ----------
        X : pandas DataFrame
            DataFrame with features
        Returns
        -------
            train_idx, test_idx: tuple numpy arrays
                indexes of resampled dataset
        """
        num_examples = X.shape[0]


        yield self._cv(np.arange(num_examples))

   