from abc import ABC, abstractmethod
import pandas as pd
import h5py
import numpy as np
import logging
class Result:
    """
    Used for passing results to the analyse results class
    """

    def __init__(self,dataset_name, strategy_name, y_true, y_pred):
        """
        Parameters
        ----------
        dataset_name: string
            Name of the dataset
        strategy_name: string
            name of the strategy
        y_true: list
            True labels
        y_pred: list
            predictions
        """
        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._y_true = y_true
        self._y_pred = y_pred

    @property
    def dataset_name(self):
        return self._dataset_name
    
    @property
    def strategy_name(self):
        return self._strategy_name
    
    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred


class MLautResult(ABC):
    @abstractmethod
    def save(self):
        """
        Saves the result
        """
    @abstractmethod
    def load(self):
        """
        method for loading the results
        """

class HDF5:
    """
    For matipulating HDF5 files
    """

    def __init__(self, hdf5_path, mode='a'):
        """
        Parameters
        ----------
        hdf5_path: string
            path to the HDF5 file. 
        mode: string
            mode for manipulating HDF5 files. Accepable values: 'r', 'r+', 'w', 'w-', 'x', 'a'
        """

        self._hdf5_path = hdf5_path
        self._mode = mode
    
    def save_dataset(self, dataset, metadata, save_path):
        """
        Saves dataset to HDF5 file

        Paramters
        ---------
        dataset: pandas DataFrame
            dataset in pandas DataFrame format
        metadata: dictionary
            metadata in the form of a dictionary
        save_path: string
            location in HDF5 database where the dataset will be saved
        """
        if 'target' not in metadata.keys():
            raise ValueError('Provide the "target" variable as a key in the metadata dictionary.')
        
        if 'dataset_name' not in metadata.keys():
            raise ValueError('Provide the "dataset_name" as a key in the metadata dictionary.')
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError('dataset must be a pandas DataFrame')
        dataset_name = metadata['dataset_name']
        store = pd.HDFStore(self._hdf5_path, self._mode)
        store[f'{save_path}/{dataset_name}'] = dataset
        store.get_storer(f'{save_path}/{dataset_name}').attrs.metadata = metadata
        store.close()

class DatasetHDF5:
    """
    Class for manipulating HDF5 data
    """
    def __init__(self, hdf5_path, dataset_path, mode='r', train_test_exists=True):
        """
        Parameters
        ----------
        hdf5_path: string
            path to the HDF5 file. 
        dataset_path: string
            Location in HDF5 database where the dataset is saved
        mode: string
            mode for manipulating HDF5 files. Accepable values: 'r', 'r+', 'w', 'w-', 'x', 'a'
        train_test_exists: Boolean
            flag whether the test train split already exists
        """
        self._hdf5_path = hdf5_path
        self._mode = mode
        self._dataset_path = dataset_path
        self._train_test_exists = train_test_exists

        self._dataset_name = self._dataset_path.split('/')[-1]

    @property
    def dataset_name(self):
        return self._dataset_path.split('/')[-1]

    def load(self):
        """
        Loads dataset from HDF5 database. The dataset needs to have been saved as a pandas Datafram originally

        Parameters
        ----------
        dataset_path: string
            path to dataset.
        return_metadata: Boolean
            Flag whether the metadata should be returned

        Returns
        -------
        (pandas DataFrame, dictionary):
            tuple with the dataset and its metadata
        """
        store = pd.HDFStore(self._hdf5_path, self._mode)
        dataset = store[self._dataset_path]
        metadata = store.get_storer(self._dataset_path).attrs.metadata

        store.close()

        self._dataset_name = metadata['dataset_name']
        return dataset, metadata

class ResultHDF5(MLautResult):

    """
    Class for saving results to a HDF5 file
    """

    def __init__(self, 
                 hdf5_path, 
                 predictions_save_path,
                 trained_strategies_save_path,
                 mode='a', 
                 experiments_predictions_group='predictions',
                 y_pred_group='y_pred',
                 y_true_group='y_true', 
                 overwrite_predictions=False):
        """
        Parameters
        ----------
        hdf5_path: string
            path to the HDF5 file. 
        predictions_save_path: string
            Path where the results are saved
        trained_strategies_save_path: string
            Path where the trained strategies will be saved
        mode: string
            mode for manipulating HDF5 files. Accepable values: 'r', 'r+', 'w', 'w-', 'x', 'a'
        experiments_predictions_group: string
            Root directory where the predictions will be saved
        y_pred_group: string
            Group for saving the predictions
        y_true_group: string
            Group for saving the true target variables
        overwrite_predictions: Boolean
            If True overwrites the predictions even if they were previously saved
        """
        self._hdf5_path = hdf5_path
        self._mode = mode
        self._predictions_save_path = predictions_save_path
        self._experiments_predictions_group = experiments_predictions_group
        self._y_true_group = y_true_group
        self._y_pred_group = y_pred_group
        self._overwrite_predictions = overwrite_predictions
        self._trained_strategies_save_path = trained_strategies_save_path
    
    @property
    def trained_strategies_save_path(self):
        return self._trained_strategies_save_path
        
    def save(self, dataset_name, strategy_name, y_true, y_pred, cv_fold):
        """
        Saves the prediction of a single trained estimator in HDF5 database.

        Parameters
        ----------
        predictions: numpy array
            array with predictions.
        dataset_name: string
            Name of dataset on which the estimator was trained.
        strategy_name: string
            name of estimator/strategy.
        """
        f = h5py.File(self._hdf5_path, self._mode)
        save_path_y_pred = f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}/{self._y_pred_group}/cv_fold{cv_fold}'
        save_path_y_true = f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}/{self._y_true_group}'

        save_path_exists = save_path_y_pred in f

        if save_path_exists and self._overwrite_predictions:
            logging.info(f'Overriding prediction for {strategy_name} on {dataset_name}.')
            del f[save_path_y_pred]
            f[save_path_y_pred] = np.array(y_pred)
        
        if not save_path_exists:
            f[save_path_y_pred] = np.array(y_pred)
        
        #TODO need to decide how to handle labels 
        f[save_path_y_true] = np.array(y_true)
        f.close()
        
    
    def load(self):
        """
        Loads dataset from HDF5 database. The dataset needs to have been saved as a numpy array originally

        Returns
        -------
        list
            List of results objects
        """
        f = h5py.File(self._hdf5_path, self._mode)

        results = []
        for dataset_name in list(f[self._experiments_predictions_group].keys()):
            for strategy_name in list(f[f'{self._experiments_predictions_group}/{dataset_name}'].keys()):
                for cv_fold in list(f[f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}/{self._y_pred_group}'].keys()):
                    #TODO handle cv folds
                    y_true = f[f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}/{self._y_true_group}'][...]
                    y_pred = f[f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}/{self._y_pred_group}/{cv_fold}'][...]
                    result = Result(dataset_name=dataset_name,
                                    strategy_name=strategy_name,
                                    y_true=y_true,
                                    y_pred=y_pred)
                    results.append(result)

        return results


  
        
    