from mlaut.estimators.mlaut_estimator import MlautEstimator
from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(BASELINE,
                                      REGRESSION, 
                                      CLASSIFICATION,
                                      GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                                      VERBOSE)

from sklearn.dummy import DummyClassifier, DummyRegressor


class Baseline_Regressor(MlautEstimator):
    """
    Wrapper for sklearn dummy regressor
    """

    def __init__(self,
                estimator=DummyRegressor(),
                hyperparameters=None,
                properties=None, 
                verbose=VERBOSE,
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):
        if properties is None:
            self._properties = {'estimator_family':[BASELINE],
                                'tasks':[REGRESSION],
                                'name':estimator.__class__.__name__}
        self._estimator = estimator
        self._hyperparameters = hyperparameters
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._num_cv_folds = num_cv_folds
        self._refit = refit

    @property
    def properties(self):
        return self._properties

    def fit(self, X, y):
        """
        Calls the estimator fit method

        Returns
        -------
        a
        """
        return self._estimator.fit(X,y)

    



class Baseline_Classifier(MlautEstimator):
    """
    Wrapper for sklearn dummy classifier class.
    """
    properties = {'estimator_family':[BASELINE],
            'tasks':[CLASSIFICATION],
            'name':'BaselineClassifier'}
    hyperparameters = None
    
    def __init__(self,
                hyperparameters=hyperparameters,
                properties=properties, 
                verbose=VERBOSE,
                n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                num_cv_folds=GRIDSEARCH_NUM_CV_FOLDS, 
                refit=True):

        self.properties = properties
        self._hyperparameters = hyperparameters
        self._verbose = verbose
        self._n_jobs = n_jobs
        self._num_cv_folds = num_cv_folds
        self._refit = refit

    def build(self, strategy='most_frequent', **kwargs):
        """
        Builds and returns estimator class.

        Parameters
        -----------
        strategy : string
            Name of strategy of baseline classifier. Default is ``most_frequent``. See `sklean.dummy.DummyClasifier <http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>`_ for additional information.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn pipeline` object
            pipeline for transforming the features and training the estimator
        """
        return self._create_pipeline(estimator=DummyClassifier(strategy=strategy))
