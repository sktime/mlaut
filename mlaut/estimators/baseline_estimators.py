from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import MlautEstimator
from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(BASELINE,
                                      REGRESSION, 
                                      CLASSIFICATION)

from sklearn.dummy import DummyClassifier, DummyRegressor
from mlaut.estimators.generic_estimator import Generic_Estimator

@properties(estimator_family=[BASELINE],
            tasks=[REGRESSION],
            name='BaselineRegressor')
class Baseline_Regressor(MlautEstimator):
    """
    Wrapper for sklearn dummy regressor
    """
    def __init__(self, verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = None

    def build(self, strategy='median', **kwargs):
        """
        Builds and returns estimator class.

        Parameters
        ----------
        strategy : string
            as per `scikit-learn dummy regressor documentation <http://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html>`_.
        **kwargs : key-value arguments.
            Ignored in this implementation. Added for compatibility with :func:`mlaut.estimators.nn_estimators.Deep_NN_Classifier`.
        
        Returns
        -------
        `sklearn.dummy.DummyRegressor` 
            Instantiated estimator object.
        """
        return DummyRegressor(strategy=strategy)
        return self._create_pipeline(estimator=DummyRegressor(strategy=strategy))

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




@properties(estimator_family=[BASELINE],
            tasks=[CLASSIFICATION],
            name='BaselineClassifier')
class Baseline_Classifier(MlautEstimator):
    """
    Wrapper for sklearn dummy classifier class.
    """
    def __init__(self, verbose=0, 
                n_jobs=-1,
                num_cv_folds=3, 
                refit=True):
        super().__init__(verbose=verbose, 
                         n_jobs=n_jobs, 
                        num_cv_folds=num_cv_folds, 
                        refit=refit)
        self._hyperparameters = None


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
