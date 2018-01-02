from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
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

def random_forest_classifier(hyperparameters=None):
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

def gaussian_naive_bayes():
    return GaussianNB()

def multinomial_naive_bayes():
    return MultinomialNB()

def bernoulli_naive_bayes():
    return BernoulliNB()


def instantiate_default_estimators(estimators, verbose=0):
    estimators_array = []

    decorator_gridsearch = gridsearch(verbose=verbose)
    
    if 'RandomForestClassifier' in estimators or 'all' in estimators:
        rfc_model = ['RandomForestClassifier', decorator_gridsearch(random_forest_classifier)()]
        estimators_array.append(rfc_model)
    
    if 'SVC' in estimators or 'all' in estimators:
        svc_model = ['SVC', decorator_gridsearch(svc)()]
        estimators_array.append(svc_model)

    if 'LogisticRegression' in estimators or 'all' in estimators:
        logisticregression_model = ['LogisticRegression', decorator_gridsearch(logisticregression)()]
        estimators_array.append(logisticregression_model)

    if 'GaussianNaiveBayes' in estimators or 'all' in estimators:
        gnb = ['GaussianNaiveBayes', gaussian_naive_bayes()]
        estimators_array.append(gnb)

    if 'BernoulliNaiveBayes' in estimators or 'all' in estimators:
        bnb = ['BernoulliNaiveBayes', bernoulli_naive_bayes()]
        estimators_array.append(bnb)

    return estimators_array