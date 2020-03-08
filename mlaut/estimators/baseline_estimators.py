from sklearn.dummy import DummyClassifier, DummyRegressor
from mlaut.estimators.base import BaseClassifier
from mlaut.estimators.base import BaseRegressor
from mlaut.highlevel.strategies import CSCStrategy


class Baseline_Classifier(BaseClassifier):
    """Wrapper for sklearn dummy classifier"""
    def __init__(self, 
                 classifier=DummyClassifier):
        self.classifier = classifier
        self.arguments = {}

class Baseline_Regressor(BaseRegressor):
    """
    Wrapper for sklearn dummy regressor
    """
    def __init__(self, 
                 regressor=DummyRegressor):
        self.regressor = regressor
        self.arguments = {}
    



