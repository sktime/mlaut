from sklearn.dummy import DummyClassifier, DummyRegressor
from sktime.classifiers.base import BaseClassifier
from sktime.regressors.base import BaseRegressor

class Baseline_Classifier(BaseClassifier):
    def __init__(self):
        self.fitted_classifier = None
    def fit(self, X, y):
        self.fitted_classifier = DummyClassifier().fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)


class Baseline_Regressor(BaseRegressor):
    """
    Wrapper for sklearn dummy regressor
    """

    def __init__(self):
        self.fitted_classifier = None
    def fit(self, X, y):
        self.fitted_classifier = DummyRegressor().fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)



