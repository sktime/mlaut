from sklearn.base import BaseEstimator

from sklearn.metrics import accuracy_score

from sktime.utils import comparison
# from sktime.utils.validation.supervised import validate_X, validate_X_y

class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """
    _estimator_type = "regressor"
    
    def fit(self, X, y):
        regressor = self.regressor(**self.arguments)
        self.fitted_regressor = regressor.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_regressor.predict(X)

class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """
    _estimator_type = "classifier"
    label_encoder = None
    random_state = None
    
    def fit(self, X, y):
        self.fitted_classifier = self.classifier.fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)
