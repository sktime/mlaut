from mlaut.estimators.mlaut_estimator import MlautEstimator
from mlaut.shared.static_variables import (GRIDSEARCH_NUM_CV_FOLDS, 
                                           GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                           VERBOSE)

class Generic_Estimator(MlautEstimator):
    """
    Generic implementation of `mlaut` estimator object. This class contains the minimum required to create an `mlaut` estimator. However, this implementation does not provide any checks and user needs to insure that sensible tuning strategies and hyperparameters are used.

    Args:
        properties_dict (dictionary): The dictionary needs to contain the following key-value pairs:

            | `estimator_family`: (array) 
            | `tasks`: (array) 
            | `name`: (string) 

            `data_preprocessing`: (dictionary)  dictionary with data preprocessing operations to be performed on datasets before they are used in training. These include `normalize_features` and `normalize_features`.
    
        estimator (scikit learn or other estimator): An instance of an estimator object. 
    """
    def __init__(self, 
                properties_dict, 
                estimator, 
                verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        #check whether all necessary properties were provided
        if 'estimator_family' not in properties_dict.keys():
            raise ValueError('Please specify estimator family in properties dictionary')
        if not isinstance(properties_dict['estimator_family'], list):
            raise ValueError('Estimator family needs to be array of strings')
        if 'name' not in properties_dict.keys():
            raise ValueError('Please specify estimator name in the properties dictionary')
        if 'tasks' not in properties_dict.keys():
            raise ValueError('Plase specify types of tasks in properties dictionary')
        if not isinstance(properties_dict['tasks'], list):
            raise ValueError('Tasks need to be an array of strings')

        self._properties_dict = properties_dict
        self._estimator = estimator
    
    def properties(self):
        return self._properties_dict
    
    def build(self, **kwargs):
        return self._estimator
