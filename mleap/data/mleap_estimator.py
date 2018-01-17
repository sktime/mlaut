from abc import ABC, abstractmethod
from ..shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS
import pickle
from ..shared.static_variables import PICKLE_EXTENTION
class MleapEstimator(ABC):
    def __init__(self, 
                verbose=0, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=3, 
                refit=True):
        self._num_cv_folds=num_cv_folds
        self._verbose=verbose
        self._n_jobs=n_jobs
        self._refit=refit

    @abstractmethod
    def build(self):
        """ Returns the estimator and its hyper parameters"""
    
    @abstractmethod
    def save(self):
        """ saves the trained model to disk """
    def load(self, path_to_model):
        #file name could be passed with .* as extention. 
        split_path = path_to_model.split('.')
        path_to_load = split_path[0] + PICKLE_EXTENTION 
        model = pickle.load(open(path_to_load,'rb'))
        self.set_trained_model(model)
    
    def set_trained_model(self, trained_model):
        self._trained_model = trained_model
    
    def get_trained_model(self):
        return self._trained_model