from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)
from mlaut.shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION
from mlaut.estimators.generic_estimator import Generic_Estimator
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB


class Gaussian_Naive_Bayes(MlautEstimator):
    """
    Wrapper for `sklearn Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`_.
    """
    properties = {'estimator_family':[NAIVE_BAYES], 
            'tasks':[CLASSIFICATION], 
            'name':'GaussianNaiveBayes'}

    def __init__(self,
                estimator=GaussianNB(),
                properties=None):
        

        if properties is None:
            self._properties = {'estimator_family':[NAIVE_BAYES], 
                                'tasks':[CLASSIFICATION], 
                                'name':estimator.__class__.__name__}
        else:
            self._properties = properties
        
        self._estimator = estimator




         


class Bernoulli_Naive_Bayes(MlautEstimator):
    """
    Wrapper for `sklearn Bernoulli Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html>`_.
    """
    def __init__(self,
                estimator=BernoulliNB(),
                properties=None):
        self._estimator = estimator

        if properties is None:
            self._properties = {'estimator_family':[NAIVE_BAYES], 
                                'tasks':[CLASSIFICATION], 
                                'name':estimator.__class__.__name__}
        else:
            self._properties = properties

        