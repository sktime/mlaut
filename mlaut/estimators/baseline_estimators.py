from mlaut.estimators.base import MlautClassifier
from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(BASELINE,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)

from sklearn.dummy import DummyClassifier, DummyRegressor
from sktime.classifiers.base import BaseClassifier

class BaselineTest(BaseClassifier):
    def __init__(self):
        self.fitted_classifier = None
    def fit(self, X, y):
        self.fitted_classifier = DummyClassifier().fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Baseline_Regressor(BaseClassifier):
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
    

class Baseline_Classifier(BaseClassifier):
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

