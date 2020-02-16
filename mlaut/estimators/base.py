from sklearn.base import BaseEstimator

from sklearn.metrics import accuracy_score

from sktime.utils import comparison
# from sktime.utils.validation.supervised import validate_X, validate_X_y

class BaseRegressor(BaseEstimator):
    """
    Base class for regressors, for identification.
    """
    _estimator_type = "regressor"

class BaseClassifier(BaseEstimator):
    """
    Base class for classifiers, for identification.
    """
    _estimator_type = "classifier"
    label_encoder = None
    random_state = None

    # def fit(self, X, y, input_checks=True):
    #     raise NotImplementedError('this is an abstract method')

    # def predict_proba(self, X, input_checks=True):
    #     raise NotImplementedError('this is an abstract method')

    # def predict(self, X, input_checks=True):
    #     """
    #     classify instances
    #     ----
    #     Parameters
    #     ----
    #     X : panda dataframe
    #         instances of the dataset
    #     input_checks : boolean
    #         whether to verify the dataset (e.g. dimensions, etc)
    #     ----
    #     Returns
    #     ----
    #     predictions : 1d numpy array
    #         array of predictions of each instance (class value)
    #     """
    #     if input_checks:
    #         validate_X(X)
    #     distributions = self.predict_proba(X, input_checks=False)
    #     predictions = []
    #     for instance_index in range(0, X.shape[0]):
    #         distribution = distributions[instance_index]
    #         prediction = comparison.arg_max(distribution, self.random_state)
    #         predictions.append(prediction)
    #     predictions = self.label_encoder.inverse_transform(predictions)
    #     return predictions

    # def score(self, X, y):
    #     validate_X_y(X, y)
    #     predictions = self.predict(X)
    #     acc = accuracy_score(y, predictions, normalize=True)
    #     return acc

class MlautClassifier(BaseClassifier):
    """
    Abstact base class that all mlaut estimators should inherit from.
    """

    _estimator_type = "classifier"
    label_encoder = None
    random_state = None

    @property
    def properties(self):
        return self._properties



    def fit(self, metadata, data):
        """
        Calls the estimator fit method

        Parameters
        ----------
        metadata: dictionary
            metadata including the target variable
        data: pandas DataFrame
            training data
        
        Returns
        -------
        sktime estimator:
            fitted estimator
        """
        y = data[metadata['target']]
        X = data.drop(metadata['target'], axis=1)
        return self._estimator.fit(X,y)    


    def predict(self, X):
        """
        Properties
        ----------
        X: dataframe or numpy array
            features on which predictions will be made
        """
        return self._estimator.predict(X)

    def save(self, dataset_name, cv_fold, strategy_save_dir):
        """
        Saves the strategy on the hard drive
        Parameters
        ----------
        dataset_name:string
            Name of the dataset
        cv_fold: int
            Number of cross validation fold on which the strategy was trained
        strategy_save_dir: string
            Path were the strategies will be saved
        """
        if strategy_save_dir is None:
            raise ValueError('Please provide a directory for saving the strategies')
        
        #TODO implement check for overwriting already saved files
        save_path = os.path.join(strategy_save_dir, dataset_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #TODO pickling will not work for all strategies
        pickle.dump(self, open(os.path.join(save_path, self._properties['name'] + '_cv_fold'+str(cv_fold)+ '.p'),"wb"))
    
    def check_strategy_exists(self, 
                              dataset_name, 
                              cv_fold,
                              strategy_save_dir):
        """
        Checks whether a strategy with the same name was already saved on the disk


        Parameters
        ----------
        dataset_name : str
            name of the dataset to check if trained
        cv_fold : int
            cv fold number
        strategy_save_dir : str
            location where the strategies are being saved
        
        Returns
        -------
        bool:
            If true strategy exists
        """
        path_to_check = os.path.join(strategy_save_dir, dataset_name, self._properties['name'] + '_cv_fold'+str(cv_fold)+ '.p')
        return os.path.exists(path_to_check)
    def load(self, path):
        """
        Load saved strategy
        Parameters
        ----------
        path: String
            location on disk where the strategy was saved
        
        Returns
        -------
        strategy:
            sktime strategy
        """
        return pickle.load(open(path,'rb'))
    

    def _create_pipeline(self, estimator):
        """
        Creates a pipeline for transforming the features of the dataset and training the selected estimator.

        Args:
            estimator (sklearn estimator): Reference of sklearn estimator that will be used at the end of the pipeline.


        Returns:
            `estimator(sklearn pipeline or GridSerachCV)`: `sklearn` pipeline object. If no preprocessing was set 
        """

        if 'data_preprocessing' in self.properties:
            data_preprocessing = self.properties['data_preprocessing']

            if data_preprocessing['normalize_labels'] is True:
                pipe = Pipeline(
                    memory=None,
                    steps=[
                        ('standardscaler', preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True) ),
                        ('estimator', estimator)
                        ]
                )
                return pipe
        else:
            return estimator