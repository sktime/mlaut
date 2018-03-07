from mleap.estimators.mleap_estimator import properties
from mleap.estimators.mleap_estimator import MleapEstimator

from mleap.shared.files_io import DiskOperations

from sklearn.model_selection import GridSearchCV
from mleap.shared.static_variables import(ENSEMBLE_METHODS, 
                                      REGRESSION, 
                                      CLASSIFICATION)
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION], 
            name='RandomForestClassifier')
class Random_Forest_Classifier(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 20, 30],
                'max_features': ['auto', 'sqrt','log2', None],
                'max_depth': [5, 15, None]
            }
        return GridSearchCV(RandomForestClassifier(), 
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
        
@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[REGRESSION], 
            name='RandomForestRegressor')
class Random_Forest_Regressor(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 20, 30],
                'max_features': ['auto', 'sqrt','log2', None],
                'max_depth': [5, 15, None]
            }
        return GridSearchCV(RandomForestRegressor(), 
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


@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION], 
            name='BaggingClassifier')
class Bagging_Classifier(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 100, 1000, 2000]
            }
        model = BaggingClassifier(base_estimator=DecisionTreeClassifier())    
        return GridSearchCV(model, 
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

@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[REGRESSION], 
            name='BaggingRegressor')
class Bagging_Regressor(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 100, 1000, 2000]
            }
        model = BaggingRegressor(base_estimator=DecisionTreeClassifier())
        return GridSearchCV(model, 
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


@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[CLASSIFICATION], 
            name='GradientBoostingClassifier')
class Gradient_Boosting_Classifier(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 100, 1000, 2000],
                'max_depth':[10,100]
            }
        return GridSearchCV(GradientBoostingClassifier(), 
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

@properties(estimator_family=[ENSEMBLE_METHODS], 
            tasks=[REGRESSION], 
            name='GradientBoostingRegressor')
class Gradient_Boosting_Regressor(MleapEstimator):

    def __init__(self, verbose=0):
        super().__init__(verbose=verbose)

    def build(self, hyperparameters=None):
        if hyperparameters is None:
            hyperparameters = {
                'n_estimators': [10, 100, 1000, 2000],
                'max_depth':[10,100]
            }
        return GridSearchCV(GradientBoostingRegressor(), 
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