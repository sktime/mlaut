from mlaut.highlevel.strategies import TabClassifStrategy, TabRegrStrategy
from sklearn.dummy import DummyClassifier, DummyRegressor

BaselineClassifierStrategy = TabClassifStrategy(estimator=DummyClassifier(), name='BaselineClassifier')
BaselineRegressorStrategy = TabRegrStrategy(estimator=DummyRegressor(), name='BaselineRegressor')