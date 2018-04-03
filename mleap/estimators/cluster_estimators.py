
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from mleap.estimators.mleap_estimator import properties
from mleap.estimators.mleap_estimator import MleapEstimator

from mleap.shared.files_io import DiskOperations
from mleap.shared.static_variables import (CLUSTER, 
                                           CLASSIFICATION,
                                           PICKLE_EXTENTION, 
                                           HDF5_EXTENTION)



@properties(estimator_family=[CLUSTER], 
            tasks=[CLASSIFICATION], 
            name='K_Means')
class K_Means(MleapEstimator):
    """
    Wrapper for `sklearn Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    """
    def save(self, dataset_name):
        """
        Saves estimator on disk.
        
        Parameters
        ----------
        dataset_name (string): name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        #set trained model method is implemented in the base class
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
    def build(self, hyperparameters=None, **kwargs):
        """
        Builds and returns estimator class.

        Parameters
        ----------
        hyperparameters (dictionary): dictionary with hyperparameters.
        kwargs(key-value): At a minimum the user must specify ``input_dim``, ``num_samples`` and ``num_classes``.

        Returns
        -------
            KMeans: `sklearn object`
        """
        input_dim=kwargs['input_dim']
        num_samples = kwargs['num_samples']
        num_classes = kwargs['num_classes']
        k_means = cluster.KMeans(n_clusters=num_classes)
        
        if hyperparameters is None:
            hyperparameters = {
                            'n_init': [10], 
                            'max_iter': [300],
                            'tol':[1e-4],
                            'n_jobs':[1] #parallelization done on GridSearchCV level
                        }
        return GridSearchCV(k_means, 
                            hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)
