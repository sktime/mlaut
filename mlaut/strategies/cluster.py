from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from mlaut.shared.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS, GRIDSEARCH_NUM_CV_FOLDS
from mlaut.highlevel.strategies import CSCStrategy
import numpy as np

hyperparameters={'n_neighbors': np.arange(1,31),
                 'p': [1, 2]
                }

estimator = GridSearchCV(neighbors.KNeighborsClassifier(), 
                                            param_grid=hyperparameters, 
                                            n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                                            cv=GRIDSEARCH_NUM_CV_FOLDS)
                                            
KNeighboursStrategy = CSCStrategy(estimator=estimator, name="KNeighbours")

