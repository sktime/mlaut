from mleap.estimators.mleap_estimator import properties
from mleap.estimators.mleap_estimator import MleapEstimator

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
    """
    Wrapper for `sklearn SVC estimator <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """
    def __init__(self, 
                verbose=0,
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        """
        calls constructor of MleapEstimator class
        """
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs,
                         num_cv_folds=num_cv_folds, 
                         refit=refit)

    def build(self, hyperparameters=None):
        """
        builds and returns estimator

        :type hyperparameters: dictionary
        :param hyperparameters: dictionary of hyperparameters to be used for tuning the estimator
        :rtype: `GridsearchCV object`
        """
        if hyperparameters is None:
            hyperparameters = {
                            'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6], #[1e-6, 1]
                            'gamma': [1e-3, 1], #[1e-3, 1e-2, 1e-1, 1]
                        }
        return GridSearchCV(SVC(), 
                            hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)


    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)


