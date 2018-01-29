from mleap.data.mleap_estimator import properties
from mleap.data.mleap_estimator import MleapEstimator

from mleap.shared.files_io import DiskOperations
from mleap.shared.static_variables import(GENERALIZED_LINEAR_MODELS,
                                      REGRESSION, 
                                      CLASSIFICATION)
from mleap.shared.static_variables import PICKLE_EXTENTION

from sklearn import linear_model
from sklearn.model_selection import GridSearchCV


@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='RidgeRegression')
class Ridge_Regression(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
       
    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'alphas':[0.1, 1, 10.0],
            
            } # this is the alpha hyperparam
        
        return linear_model.RidgeCV(alphas=hyperparameters['alphas'],
                                cv=self._num_cv_folds)
    def save(self, dataset_name):
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='Lasso')
class Lasso(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
    
    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'alphas':[0.1, 1, 10.0]}
        return linear_model.LassoCV(alphas=hyperparameters['alphas'],
                                    cv=self._num_cv_folds)
    def save(self, dataset_name):
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)
@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='LassoLars')
class Lasso_Lars(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
    

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {'max_n_alphas':1000}
        return linear_model.LassoLarsCV(max_n_alphas=hyperparameters['max_n_alphas'],
                                    cv=self._num_cv_folds)
    def save(self, dataset_name):
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=self._trained_model,
                                model_name=self.properties()['name'],
                                dataset_name=dataset_name)

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS], 
            tasks=[CLASSIFICATION,REGRESSION], 
            name='LogisticRegression')
class Logistic_Regression(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
            }
        return GridSearchCV(linear_model.LogisticRegression(), 
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

@properties(estimator_family=[GENERALIZED_LINEAR_MODELS],
            tasks=[CLASSIFICATION],
            name='PassiveAggressiveClassifier')
class Passive_Aggressive_Classifier(MleapEstimator):
    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)
    def build(self, hyperparameters=None):
        hyperparameters = {
                'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6],
                'max_iter':[1000]
            }
        return GridSearchCV(linear_model.PassiveAggressiveClassifier(), 
                            hyperparameters, 
                            verbose=self._verbose
                            )

    
    def save(self, dataset_name):
        trained_model = self._trained_model
        disk_op = DiskOperations()
        disk_op.save_to_pickle(trained_model=trained_model,
                             model_name=self.properties()['name'],
                             dataset_name=dataset_name)
