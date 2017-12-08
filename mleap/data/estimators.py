from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

'''
See for tutorial on decorators with parameters
http://www.artima.com/weblogs/viewpost.jsp?thread=240845
https://stackoverflow.com/questions/5929107/decorators-with-parameters
'''
def gridsearch(verbose=1, n_jobs=1, refit=True):
    def real_decorator(function):
        def wrapper(*args):
            #the argumets are passed as 
            model, hyperparameters = function()
            gs = GridSearchCV(model, hyperparameters, verbose=verbose, n_jobs=n_jobs, refit=refit)
            return gs
        return wrapper
    return real_decorator

def randomforestclassifier(hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': [10, 20, 30],
            'max_features': ['auto', 'sqrt','log2', None],
            'max_depth': [5, 15, None]
        }
    return RandomForestClassifier(), hyperparameters

def svc(hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
                        'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
                        'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
                    }
    return SVC(), hyperparameters

def logisticregression(hyperparameters=None):
    if hyperparameters is None:
        hyperparameters = {
            'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
        }
    return linear_model.LogisticRegression(), hyperparameters


def instantiate_default_estimators():
    decorator_gridsearch = gridsearch(verbose=0)
    rfc_model = ['RandomForestClassifier', decorator_gridsearch(randomforestclassifier)()]
    svc_model = ['SVM', decorator_gridsearch(svc)()]
    logisticregression_model = ['LogisticRegression', decorator_gridsearch(logisticregression)()]

    return [rfc_model, svc_model, logisticregression_model]