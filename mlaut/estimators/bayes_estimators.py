
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from sktime.classifiers.base import BaseClassifier
from sktime.regressors.base import BaseRegressor

class Gaussian_Naive_Bayes(BaseClassifier):
    """
    Wrapper for `sklearn Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`_.
    """
    def __init__(self):
        self.fitted_classifier = None
    def fit(self, X, y):
        self.fitted_classifier = GaussianNB().fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)

class Bernoulli_Naive_Bayes(BaseRegressor):
    """
    Wrapper for `sklearn Bernoulli Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html>`_.
    """
    def __init__(self):
        self.fitted_classifier = None
    def fit(self, X, y):
        self.fitted_classifier = BernoulliNB().fit(X,y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        return self.fitted_classifier.predict(X)