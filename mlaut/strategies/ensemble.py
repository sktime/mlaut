from mlaut.highlevel.strategies import TabClassifStrategy, TabRegrStrategy
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import(GRIDSEARCH_NUM_CV_FOLDS,
                                      GRIDSEARCH_CV_NUM_PARALLEL_JOBS)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import numpy as np

max_depth = {
    'RandomForest':[10,100, None],
    'Boosting':  np.arange(1,11)
}
max_features= {
    'RandomForest':['auto', 'sqrt','log2', None],
    'Bagging':[0.5,1]
}
min_samples_split = [2, 3, 10]
bootstrap = [True, False]
criterion = ["gini", "entropy"]
n_estimators = [10, 100, 200, 500]
base_estimator = {
    'classifier': [DecisionTreeClassifier()],
    'regressor': [DecisionTreeRegressor()]
}
max_samples = [0.5, 1]
learning_rate= [0.01, 0.1, 1, 10, 100]


rf_classifier = GridSearchCV(RandomForestClassifier(), 
                            param_grid={
                                'max_depth': max_depth['RandomForest'],
                                'max_features': max_features['RandomForest'],
                                'min_samples_split':min_samples_split,
                                'bootstrap': bootstrap,
                                'criterion': criterion,
                                'n_estimators': n_estimators
                            }, 
                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                            cv=GRIDSEARCH_NUM_CV_FOLDS)

RandomForestClassifierStrategy = TabClassifStrategy(estimator=rf_classifier, name='RandomForestClassifier')

rf_regressor = GridSearchCV(RandomForestRegressor(), 
                            param_grid={
                                'max_depth':max_depth['RandomForest'],
                                'max_features': max_features['RandomForest'],
                                'n_estimators':n_estimators
                            }, 
                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                            cv=GRIDSEARCH_NUM_CV_FOLDS)

RandomForestRegressorStrategy = TabRegrStrategy(estimator=rf_regressor, name='RandomForestRegressor')


bagging_classifier = GridSearchCV(BaggingClassifier(), 
                            param_grid={
                                'max_features': max_features['Bagging'],
                                'n_estimators': n_estimators,
                                'max_samples': max_samples,
                                'base_estimator': base_estimator['classifier']
                            }, 
                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                            cv=GRIDSEARCH_NUM_CV_FOLDS)

BaggingClassifierStrategy = TabClassifStrategy(estimator=bagging_classifier, name='BaggingClassifier')

bagging_regressor = GridSearchCV(BaggingRegressor(), 
                            param_grid={
                                'max_features': max_features['Bagging'],
                                'n_estimators':n_estimators,
                                'max_samples': max_samples,
                                'base_estimator': base_estimator['regressor']
                            }, 
                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                            cv=GRIDSEARCH_NUM_CV_FOLDS)

BaggingRegressorStrategy = TabRegrStrategy(estimator=bagging_regressor, name='BaggingRegressor')


gradient_boosting_classifier = GridSearchCV(GradientBoostingClassifier(), 
                            param_grid={
                                'n_estimators':n_estimators,
                                'max_depth': max_depth['Boosting'],
                                'learning_rate': learning_rate
                            }, 
                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                            cv=GRIDSEARCH_NUM_CV_FOLDS)
GradientBoostingClassifierStrategy = TabClassifStrategy(estimator=gradient_boosting_classifier, name='GradientBoostingClassifier')

gradient_boosting_regressor =  GridSearchCV(GradientBoostingRegressor(), 
                            param_grid={
                                'n_estimators':n_estimators,
                                'max_depth': max_depth['Boosting'],
                                'learning_rate': learning_rate
                            }, 
                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS,
                            cv=GRIDSEARCH_NUM_CV_FOLDS)
GradientBoostingRegressorStrategy = TabClassifStrategy(estimator=gradient_boosting_regressor, name='GradientBoostingRegressor')
