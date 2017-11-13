from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from src.static_variables import GRIDSEARCH_CV_NUM_PARALLEL_JOBS
from sklearn.svm import SVC

class ModelsContainer(object):
    '''
    Sets up the models that will be applied in all experiments
    '''
    def create_models(self):
        models = []
        clf_rfc = RandomForestClassifier()
        rfc_params = {
            'n_estimators': [10, 20, 30],
            'max_features': ['auto', 'sqrt','log2', None],
            'max_depth': [5, 15, None]
        }
        gs_random_Forest = GridSearchCV(clf_rfc, rfc_params, verbose=1, 
            refit=True, n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS)
        models.append(['RandomForestClassifier', gs_random_Forest])

        clf_svm = SVC()
        svm_params = {
            'C': [1e-6, 1], #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
            'gamma': [1e-3, 1] #[1e-3, 1e-2, 1e-1, 1]
        }
        gs_svm = GridSearchCV(clf_svm, svm_params)
        gs_svm = GridSearchCV(clf_svm, svm_params, verbose=1, 
            refit=True, n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS)      
        models.append(['SVM', gs_svm])

        clf_logisticReg = linear_model.LogisticRegression()
        logisticReg_params = {
            'C': [1e-6, 1] #[1e-6, 1e-5, 1e-4,1e-3, 1e-2, 1, 1e2,1e3,1e4,1e5,1e6]
        }
        gs_logistic_reg = GridSearchCV(clf_logisticReg, logisticReg_params, 
            verbose=1, refit=True, n_jobs=GRIDSEARCH_CV_NUM_PARALLEL_JOBS)
        models.append(['LogisticRegression', gs_logistic_reg])

        return models
