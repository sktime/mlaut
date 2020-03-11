from mlaut.shared.static_variables import GRIDSEARCH_NUM_CV_FOLDS, GRIDSEARCH_CV_NUM_PARALLEL_JOBS
from mlaut.highlevel.strategies import CSCStrategy, CSRStrategy
import numpy as np
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV

c_range = np.logspace(-2, 10, 13)
gamma_range = np.append(np.logspace(-9, 3, 13),'scale')
kernel_rbf='rbf'
kernel_linear='linear'

svm_estimator = GridSearchCV(estimator=SVC(),
                        param_grid=[
                            {
                                'kernel':kernel_rbf,
                                'gamma':gamma_range,
                                'C': c_range
                            },
                            {
                                'kernel': kernel_linear,
                                'C':c_range
                            }
                        ],
                        n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                        cv=GRIDSEARCH_NUM_CV_FOLDS)

SVMStrategy = CSCStrategy(estimator=svm_estimator, name='SVM')

svr_estimator = GridSearchCV(estimator=SVR(),
                        param_grid=[
                            {
                                'kernel':kernel_rbf,
                                'gamma':gamma_range,
                                'C': c_range
                            },
                            {
                                'kernel': kernel_linear,
                                'C':c_range
                            }
                        ],
                        n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS, 
                        cv=GRIDSEARCH_NUM_CV_FOLDS)

SVRStrategy = CSRStrategy(estimator=svr_estimator, name='SVR')

