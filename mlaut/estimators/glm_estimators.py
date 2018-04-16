from mlaut.estimators.mlaut_estimator import properties
from mlaut.estimators.mlaut_estimator import mlautEstimator

from mlaut.shared.files_io import DiskOperations
from mlaut.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      REGRESSION, 
                                      CLASSIFICATION)
from mlaut.shared.static_variables import PICKLE_EXTENTION

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='RidgeRegression')
class Ridge_Regression(mlautEstimator):
    """
    Wrapper for `sklearn Ridge Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_.
    """
    def __init__(self):
        super().__init__()
        self._hyperparameters = {'alphas':[0.1, 1, 10.0],
            
            } # this is the alpha hyperparam
       
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
        `sklearn object with built-in cross-validation`
            Instantiated estimator object.
        """
        
        
        return linear_model.RidgeCV(alphas=self._hyperparameters['alphas'],
                                cv=self._num_cv_folds)
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='Lasso')
class Lasso(mlautEstimator):
    """
    Wrapper for `sklearn Lasso <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html>`_.
    """
    def __init__(self):
        super().__init__()
        self._hyperparameters = {'alphas':[0.1, 1, 10.0]}
    
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
        `sklearn object with built-in cross-validation`
            Instantiated estimator object.
        """

        return linear_model.LassoCV(alphas= self._hyperparameters['alphas'],
                                    cv=self._num_cv_folds)
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='LassoLars')
class Lasso_Lars(mlautEstimator):
    """
    Wrapper for `sklearn Lasso Lars <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html>`_.
    """

    
    def __init__(self):
        super().__init__()
        self._hyperparameters = {'max_n_alphas':1000}
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
        `sklearn object with built-in cross-validation`
            Instantiated estimator object.
        """

        return linear_model.LassoLarsCV(max_n_alphas=self._hyperparameters['max_n_alphas'],
                                    cv=self._num_cv_folds)
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[REGRESSION], 
            name='LogisticRegression')
class Logistic_Regression(mlautEstimator):
    """
    Wrapper for `sklearn Logistic Regression <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
    """
    def __init__(self):
        super().__init__()
        self._hyperparameters = {
                'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
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

        return GridSearchCV(linear_model.LogisticRegression(), 
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

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS],
            tasks=[CLASSIFICATION],
            name='PassiveAggressiveClassifier')
class Passive_Aggressive_Classifier(mlautEstimator):
    """
    Wrapper for `sklearn Passive Aggressive Classifier <http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html>`_.
    """

    def __init__(self):
        super().__init__()
        self._hyperparameters = {
                'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6],
                'max_iter':[1000]
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
        `GridsearchCV object`
            Instantiated estimator object.
        """
 
        return GridSearchCV(linear_model.PassiveAggressiveClassifier(), 
                            self._hyperparameters, 
                            verbose=self._verbose
                            )

    
    def save(self, dataset_name):
        """
        Saves estimator on disk.

        :type dataset_name: string
        :param dataset_name: name of the dataset. Estimator will be saved under default folder structure `/data/trained_models/<dataset name>/<model name>`
        """
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
