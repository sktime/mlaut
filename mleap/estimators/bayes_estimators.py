from mleap.estimators.mleap_estimator import properties
from mleap.estimators.mleap_estimator import MleapEstimator

from mleap.shared.files_io import DiskOperations
from mleap.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      ENSEMBLE_METHODS, 
                                      SVM,
                                      NEURAL_NETWORKS,
                                      NAIVE_BAYES,
                                      REGRESSION, 
                                      CLASSIFICATION)
from mleap.shared.static_variables import PICKLE_EXTENTION, HDF5_EXTENTION

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

@properties(estimator_family=[NAIVE_BAYES], 
            tasks=[CLASSIFICATION], 
            name='GaussianNaiveBayes')
class Gaussian_Naive_Bayes(MleapEstimator):


    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
    def build(self):
        return GaussianNB()

@properties(estimator_family=[NAIVE_BAYES], 
            tasks=[CLASSIFICATION], 
            name='BernoulliNaiveBayes')
class Bernoulli_Naive_Bayes(MleapEstimator):

    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
    def build(self):
        return BernoulliNB()
