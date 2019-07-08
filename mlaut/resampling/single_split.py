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

    def __init__(self, cv, random_state=None):
        """
        Parameters
        ----------
        cv: sktime.model_selection object
            split object
        random_state : int
            state for random seed
        """
        self._cv = cv
        self._random_state=random_state
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


        yield self._cv(np.arange(num_examples), random_state=self._random_state)

   