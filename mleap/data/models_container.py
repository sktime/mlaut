from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

class ModelsContainer(object):
    '''
    Sets up the models that will be applied in all experiments
    '''
    def instantiate_models(self, verbose=None, num_parallel_jobs = None, **kwargs):
        models = []
        if verbose is None:
            verbose = 1
        if num_parallel_jobs is None:
            num_parallel_jobs = 1
            
        for strategy in kwargs:
            if strategy == 'RandomForestClassifier':
                clf_rfc = RandomForestClassifier()
                if kwargs[strategy] is not None:
                    rfc_params = kwargs[strategy]
                else:
                    rfc_params = {
                    'n_estimators': [10, 20, 30],
                    'max_features': ['auto', 'sqrt','log2', None],
                    'max_depth': [5, 15, None]
                    }
                gs_random_Forest = GridSearchCV(clf_rfc, rfc_params, verbose=verbose, 
                    refit=True, n_jobs=num_parallel_jobs)
                models.append([strategy, gs_random_Forest])
            
            if strategy == 'SVM':
                clf_svm = SVC()
                if kwargs[strategy] is not None:
                    svm_params = kwargs[strategy]
                else:
                    svm_params = {
                        'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                        'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
                    }
                gs_svm = GridSearchCV(clf_svm, svm_params, 
                    verbose=verbose, 
                    refit=True, n_jobs=num_parallel_jobs)      
                models.append([strategy, gs_svm])
            
            if strategy == 'LogisticRegression':
                clf_logisticReg = linear_model.LogisticRegression()
                if kwargs[strategy] is not None:
                    logisticReg_params = kwargs[strategy]
                else:
                    logisticReg_params = {
                        'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                    }
                gs_logistic_reg = GridSearchCV(clf_logisticReg, 
                        logisticReg_params, 
                        verbose=verbose, 
                            refit=True, n_jobs=num_parallel_jobs)
                models.append([strategy, gs_logistic_reg])
        return models

    def create_models(self, n_jobs=None, **kwargs):
        models = []
        clf_rfc = RandomForestClassifier()
        rfc_params = {
            'n_estimators': [10, 20, 30],
            'max_features': ['auto', 'sqrt','log2', None],
            'max_depth': [5, 15, None]
        }
        gs_random_Forest = GridSearchCV(clf_rfc, rfc_params, verbose=1, 
            refit=True, n_jobs=n_jobs)
        models.append(['RandomForestClassifier', gs_random_Forest])

        clf_svm = SVC()
        svm_params = {
            'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
            'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
        }
        gs_svm = GridSearchCV(clf_svm, svm_params, verbose=1, 
            refit=True, n_jobs=n_jobs)      
        models.append(['SVM', gs_svm])

        clf_logisticReg = linear_model.LogisticRegression()
        logisticReg_params = {
            'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
        }
        gs_logistic_reg = GridSearchCV(clf_logisticReg, logisticReg_params, 
            verbose=1, refit=True, n_jobs=n_jobs)
        models.append(['LogisticRegression', gs_logistic_reg])

        return models
