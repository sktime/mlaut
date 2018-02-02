from mleap.data.mleap_estimator import properties
from mleap.data.mleap_estimator import MleapEstimator

from mleap.shared.files_io import DiskOperations
from sklearn.model_selection import GridSearchCV
from mleap.shared.static_variables import(SVM,
                                      REGRESSION, 
                                      CLASSIFICATION)

from sklearn.svm import SVC

@properties(estimator_family=[SVM], 
            tasks=[CLASSIFICATION], 
            name='SVC')
class SVC_mleap(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                            'C': [1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6], #[1e-6, 1]
                            'gamma': [1e-3, 1e-2, 1e-1, 1] #[1e-3, 1]
                        }
        return GridSearchCV(SVC(), 
                            hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)


    def save(self, dataset_name):
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)


