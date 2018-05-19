
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import MlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import (CLUSTER, 
                                           CLASSIFICATION,
                                           PICKLE_EXTENTION, 
                                           HDF5_EXTENTION)



@properties(estimator_family=[CLUSTER], 
            tasks=[CLASSIFICATION], 
            name='K_Means')
class K_Means(MlautEstimator):
    """
    Wrapper for `sklearn Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_.
    """
    def __init__(self, verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = {
                                'n_init': [10], 
                                'max_iter': [300],
                                'tol':[1e-4],
                                'n_jobs':[1] #parallelization done on GridSearchCV level
                        }
    # def save(self, dataset_name):
    #     """
    #     Saves estimator on disk.
        
    #     Parameters
    #     ----------
    #     dataset_name (string): name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
    #     """
    #     #set trained model method is implemented in the base class
    #     trained_model = self._trained_model
    #     disk_op = DiskOperations()
    #     disk_op.save_to_pickle(trained_model=trained_model,
    #                          model_name=self.properties()['name'],
    #                          dataset_name=dataset_name)
    def build(self, **kwargs):
        """
        Builds and returns estimator class.

        Parameters
        ----------
        hyperparameters (dictionary): dictionary with hyperparameters.
        kwargs(key-value): At a minimum the user must specify ``input_dim``, ``num_samples`` and ``num_classes``.

        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
        # input_dim=kwargs['input_dim']
        # num_samples = kwargs['num_samples']
        num_classes = kwargs['num_classes']
        k_means = cluster.KMeans(n_clusters=num_classes)
        

        estimator = GridSearchCV(k_means, 
                            self._hyperparameters, 
                            verbose = self._verbose,
                            n_jobs=self._n_jobs,
                            refit=self._refit)
        return self._create_pipeline(estimator=estimator)        

