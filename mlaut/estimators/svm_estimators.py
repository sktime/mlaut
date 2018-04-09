from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import mlautEstimator

from mlaut.shared.files_io import DiskOperations
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import(SVM,
                                      REGRESSION, 
                                      CLASSIFICATION)

from sklearn.svm import SVC

@properties(estimator_family=[SVM], 
            tasks=[CLASSIFICATION], 
            name='SVC')
class SVC_mlaut(mlautEstimator):
    """
    Wrapper for `sklearn SVC estimator <http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.
    """
    def __init__(self):
        super().__init__()
        self._hyperparameters = {
                            'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6], #[1e-6, 1]
                            'gamma': [1e-3, 1], #[1e-3, 1e-2, 1e-1, 1]
                        }

    def build(self, **kwargs):
        """
        builds and returns estimator
        
        Parameters
        ----------
        hyperparameters: dictionary
            Dictionary of hyperparameters to be used for tuning the estimator.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `GridsearchCV` object
            Instantiated estimator object.
        
        """

        return GridSearchCV(SVC(), 
                            self._hyperparameters, 
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


