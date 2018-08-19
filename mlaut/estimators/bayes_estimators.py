from mlaut.estimators.mlaut_estimator import properties
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

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB

@properties(estimator_family=[NAIVE_BAYES], 
            tasks=[CLASSIFICATION], 
            name='GaussianNaiveBayes')
class Gaussian_Naive_Bayes(MlautEstimator):
    """
    Wrapper for `sklearn Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html>`_.
    """
    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = None

   
    def build(self, **kwargs):
        """
        Builds and returns estimator class.
        
        Parameters
        ----------
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
        return self._create_pipeline(estimator=GaussianNB())        

         

@properties(estimator_family=[NAIVE_BAYES], 
            tasks=[CLASSIFICATION], 
            name='BernoulliNaiveBayes')
class Bernoulli_Naive_Bayes(MlautEstimator):
    """
    Wrapper for `sklearn Bernoulli Naive Bayes estimator <http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html>`_.
    """
    def __init__(self, verbose=VERBOSE, 
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = None
    

    # def save(self, dataset_name):
    #     """
    #     Saves estimator on disk.

    #     :type dataset_name: string
    #     :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
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
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
        return self._create_pipeline(estimator=BernoulliNB())
