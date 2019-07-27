from mlaut.estimators.mlaut_estimator import MlautEstimator
from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(BASELINE,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)

from sklearn.dummy import DummyClassifier, DummyRegressor


class Baseline_Regressor(MlautEstimator):
    """
    Wrapper for sklearn dummy regressor
    """

    def __init__(self,
                estimator=DummyRegressor(),
                properties=None):
        if properties is None:
            properties = {'estimator_family':[BASELINE],
                                'tasks':[REGRESSION],
                                'name':estimator.__class__.__name__}
        
        self._estimator = estimator
        self._properties = properties
    

class Baseline_Classifier(MlautEstimator):
    """
    Wrapper for sklearn dummy regressor
    """

    def __init__(self,
                estimator=DummyClassifier(),
                hyperparameters=None,
                properties=None, 
                verbose=VERBOSE):
        if properties is None:
            properties = {'estimator_family':[BASELINE],
                                'tasks':[CLASSIFICATION],
                                'name':estimator.__class__.__name__}


        self._estimator = estimator
        self._properties = properties
