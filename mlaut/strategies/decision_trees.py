from mlaut.highlevel.strategies import CSCStrategy, CSRStrategy
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS, GRIDSEARCH_NUM_CV_FOLDS


max_depth = [10,100, None]
criterion = {
    'classifier': ['gini', 'entropy'],
    'regressor': ['mse', 'friedman_mse', 'mae']
}
max_features = ['auto', 'sqrt','log2']
min_samples_leaf = np.arange(1,11)

classifier = GridSearchCV(DecisionTreeClassifier(), 
                        param_grid={
                            'max_depth': max_depth,
                            'criterion': criterion['classifier'],
                            'max_features': max_features,
                            'min_samples_leaf': min_samples_leaf
                        }, 
                        n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                        cv=GRIDSEARCH_NUM_CV_FOLDS)

DecisionTreeClassifierStrategy = CSCStrategy(estimator=classifier, name="DecisionTreeClassifier")

regressor = GridSearchCV(DecisionTreeRegressor(), 
                        param_grid={
                            'max_depth': max_depth,
                            'criterion': criterion['regressor'],
                            'max_features': max_features,
                            'min_samples_leaf': min_samples_leaf
                        }, 
                        n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                        cv=GRIDSEARCH_NUM_CV_FOLDS)


DecisionTreeRegressorStrategy = CSRStrategy(estimator=regressor, name="DecisionTreeRegressor")

