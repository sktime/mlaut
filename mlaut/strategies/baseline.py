from mlaut.highlevel.strategies import CSCStrategy, CSRStrategy
from sklearn.dummy import DummyClassifier, DummyRegressor

BaselineClassifier = CSCStrategy(estimator=DummyClassifier(), name='BaselineClassifier')
BaselineRegressor = CSRStrategy(estimator=DummyRegressor(), name='BaselineRegressor')