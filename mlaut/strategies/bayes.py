from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from mlaut.highlevel.strategies import CSCStrategy, CSRStrategy

GaussianNaiveBayesStrategy = CSCStrategy(GaussianNB(), name="GaussianNaiveBayes")
BernoulliNaiveBayesStrategy = CSRStrategy(BernoulliNB(), name="BernoulliNB")