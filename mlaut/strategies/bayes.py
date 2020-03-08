from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

from mlaut.highlevel.strategies import CSCStrategy, CSRStrategy

GaussianNaiveBayes = CSCStrategy(GaussianNB(), name="GaussianNaiveBayes")
BernoulliNaiveBayes = CSRStrategy(BernoulliNB(), name="BernoulliNB")