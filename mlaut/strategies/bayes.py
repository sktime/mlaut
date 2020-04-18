from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from mlaut.highlevel.strategies import TabClassifStrategy, TabRegrStrategy

GaussianNaiveBayesStrategy = TabClassifStrategy(GaussianNB(), name="GaussianNaiveBayes")
BernoulliNaiveBayesStrategy = TabRegrStrategy(BernoulliNB(), name="BernoulliNB")