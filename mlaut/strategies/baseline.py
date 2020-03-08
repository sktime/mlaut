from mlaut.highlevel.strategies import CSCStrategy, CSRStrategy
from sklearn.dummy import DummyClassifier, DummyRegressor

BaselineClassifierStrategy = CSCStrategy(estimator=DummyClassifier(), name='BaselineClassifier')
BaselineRegressorStrategy = CSRStrategy(estimator=DummyRegressor(), name='BaselineRegressor')