import os
import glob
import pickle
from .static_variables import (EXPERIMENTS_TRAINED_MODELS_DIR, 
                               EXPERIMENTS_PREDICTIONS_GROUP, 
                               EXPERIMENTS_MODEL_ACCURACY_DIR, 
                               PICKLE_EXTENTION, 
                               HDF5_EXTENTION, 
                               REFORMATTED_DATASETS_DIR,
                               RUNTIMES_GROUP,
                               RESULTS_DIR)

import h5py
import tables
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

class DiskOperations(object):
    """
    Class with high level disk operations for loading and saving estimators.
    """

    def save_to_pickle(self, 
                       trained_model, 
                       model_name, 
                       dataset_name, 
                       root_dir=EXPERIMENTS_TRAINED_MODELS_DIR):
        """load_dataset_pd
        Saves sklearn estimator to disk as pickle file.

        Args:
            trained_model(sklearn estimator object): trained sklearn object to be saved on disk.
            model_name(string): name of sklearn estimator.
            dataset_name(string): name of dataset that the estimator was trained on.
            root_dir(string): root dir where the trained estimators will be saved. 
        """
        if not os.path.exists(root_dir + os.sep + dataset_name):
            os.makedirs(root_dir + os.sep + dataset_name)
        
        with open(root_dir + os.sep + dataset_name + os.sep + model_name + PICKLE_EXTENTION, 'wb') as f:
            pickle.dump(trained_model,f)

    def save_keras_model(self, 
                        trained_model, 
                        model_name, 
                        dataset_name, 
                        root_dir=EXPERIMENTS_TRAINED_MODELS_DIR):
        """
        Saves keras object to disk.

        :type trained_model: keras estimator object.
        :param trained_model: trained keras object to be saved on disk.

        :type model_name: string
        :param model_name: name of sklearn estimator.

        :type dataset_name: string
        :param dataset_name: name of dataset that the estimator was trained on.

        :type root_dir: string
        :param root_dir: root dir where the trained estimators will be saved. 
        """
        if not os.path.exists(root_dir + os.sep + dataset_name):
            os.makedirs(root_dir + os.sep + dataset_name)
        
        trained_model.model.save(root_dir + os.sep + dataset_name + os.sep + model_name + HDF5_EXTENTION)
    
    def check_path_exists(self, path_to_file):
        """
        Checks whether path exists on disk
        Args:
            path_to_file (string): path to check whether exists or not 
        Returns:
            (Bolean, String): A pickle the first element indicates whether the path exists and the second element returns the actual file path.
        """
        path_exists = glob.glob(path_to_file)
        if len(path_exists) == 1:
            return (True, path_exists[0])
        elif len(path_exists) > 1:
            raise ValueError(f'Multiple files found: {path_to_file}.')
        else:
            return (False, None)

    def create_directory_on_hdd(self, directory_path):
        """
        Creates directory on hard drive

        :type directory_path: string.
        :param directory_path: path of directory that will be created.
        """
        os.makedirs(directory_path, exist_ok=True)
class FilesIO:
    """
    Methods for manipulating HDF5 databases and datasets.

    :type hdf5_filename: string
    :param hdf5_filename: full path where the database file will be stored.

    :type mode: string
    :param mode: open and create file modes as per the `h5py documentation <http://docs.h5py.org/en/latest/high/file.html>`_.
    
    :type experiments_predictions_group: string
    :param experiments_predictions_group: Location in HDF5 database where estimator predictions will be saved.
    """
    def __init__(self, hdf5_filename, mode='a', 
                 experiments_predictions_group=EXPERIMENTS_PREDICTIONS_GROUP):
        self._hdf5_filename = hdf5_filename
        self._mode = mode
        self._experiments_predictions_group=experiments_predictions_group


    def check_h5_path_exists(self, path_to_check):
        """
        Checks whether path/group exists in HDF5 database

        :type path_to_check: string
        :param path_to_check: path of group that will be checked.
        :rtype: `boolean`
        """
        f = h5py.File(self._hdf5_filename, self._mode)
        is_present = path_to_check in f
        f.close()
        return is_present 

    def load_predictions_for_dataset(self, dataset_name):
        """
        Loads predictions generated my trained estimator models.

        Args:
            dataset_name(string): Name of dataset on which estimators were trained
        Retuns:
            predictions_for_dataset(`array): Array in the form [[strategy name][predictions]]`.
        """
        f = h5py.File(self._hdf5_filename, self._mode)
        load_path = f'/{self._experiments_predictions_group}/{dataset_name}'
        predictions = f[load_path]
        
        
        predictions_for_dataset = []
        for strategy in list(predictions):
            predictions_for_dataset.append([strategy, predictions[strategy][...]])
        f.close()
        return predictions_for_dataset


    # def save_trained_models_to_disk(self, trained_models, dataset_name):
    #     with open(EXPERIMENTS_TRAINED_MODELS_DIR + dataset_name + PICKLE_EXTENTION,'wb') as f:
    #         pickle.dump(trained_models,f)
    
    def save_prediction_to_db(self, predictions, dataset_name, strategy_name):
        """
        Saves the prediction of a single trained estimator in HDF5 database.

        Args:
            predictions(numpy array): array with predictions.
            dataset_name(string): dataset_name: name of dataset on which the estimator was trained.
            strategy_name(string):strategy_name: name of estimator/strategy.
        """
        f = h5py.File(self._hdf5_filename, self._mode)
        save_path = f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}'
        try:
            save_path_exists = save_path in f
            if save_path_exists:
                del f[save_path]
                logging.info(f'Overriding prediction for {strategy_name} on {dataset_name}.')
            f[save_path] = np.array(predictions)
            logging.info(f'Prediction of {strategy_name} on {dataset_name} stored in database.') 
        except Exception as e:
            raise ValueError(f'Exception occurred: {e}')
        finally:
            f.close()

    def save_predictions_to_db(self, predictions, dataset_name):
        """
        Bulk save of predictions generated by a several trained estimators.

        :type predictions: array 
        :param predictions: array in form [[estimator name][predictions]]

        :type dataset_name: string
        :param dataset_name: Name of dataset on which the estimator was trained.
        """
        # TODO this seems a duplicate to def save_numpy_array_hdf5
        f = h5py.File(self._hdf5_filename, self._mode)
        for prediction in predictions:
            strategy_name = prediction[0]
            strategy_predictions = np.array(prediction[1])
            save_path = f'{self._experiments_predictions_group}/{dataset_name}/{strategy_name}'
            try:
                f[save_path] = strategy_predictions
                logging.info(f'Predictions of {strategy_name} on {dataset_name} stored in database.')
            except Exception as e:
                raise ValueError(f'Exception while saving: {e}')
        f.close()
    def save_array_hdf5(self, group, datasets, array_names, array_meta):
        """
        Saves an array of data to HDF5 database

        :type group:string
        :param group: path to group in HDF5 where the arrays will be saved

        :type datasets: array of arrays
        :param datasets: array in the form [[x_1],[x_2],...[x_n]] representing the data that will be saved. Each x_i is saved
                         in a different sub-group under the common parent HDF5 group.
        :type array_names: array of strings
        :param array_names: name of sub-groups where the dataset arrays will be saved

        :type array_meta: array of dictionaries
        :param array_meta: metadata for each sub-group 
        """
        #TODO metadata not saved
        #create groups
        f = h5py.File(self._hdf5_filename, self._mode)
        f.create_group(group)
        for dts in zip(datasets, array_names, array_meta):
            data = dts[0]
            name = dts[1]
            meta = dts[2]
            f[group + '/' + name] = data
            #save metadata
            for k in meta.keys():
                f[group + '/' + name].attrs[k] = meta[k]
        f.close()
            
    # def save_prediction_accuracies_to_db(self, model_accuracies):
    #     for model in model_accuracies:
    #         model_name = model[0]
    #         model_accuracy = np.array([model[1]])
            
    #         #create group if necessary
    #         f =  h5py.File(self.hdf5_filename, self._mode)
    #         if not '/' + EXPERIMENTS_MODEL_ACCURACY_DIR in f:
    #             f.create_group('/' + EXPERIMENTS_MODEL_ACCURACY_DIR)
    #         f.close()            
    #         #create array with accuracies or append it
    #         f = tables.open_file(self.hdf5_filename, self._mode)
    #         if not  '/' + EXPERIMENTS_MODEL_ACCURACY_DIR + model_name in f: 
    #             f.create_earray('/' +EXPERIMENTS_MODEL_ACCURACY_DIR, name=model_name, obj=model_accuracy)
    #         else:
    #             mdl_acc = getattr(f.root.experiments.trained_models_accuracies, model_name)
    #             mdl_acc.append(model_accuracy)
    #         f.close()
            
    # def get_prediction_accuracies_per_strategy(self):
    #     pred_accuracies = {}
    #     f = h5py.File(self.hdf5_filename, self._mode)
    #     strategies = f[EXPERIMENTS_MODEL_ACCURACY_DIR]
    #     for strategy in strategies:
    #         pred_accuracies[strategy] = strategies[strategy][...]
    #     return pred_accuracies
    
    def save_ml_strategy_timestamps(self, timestamps_df, dataset_name, overwrite_timestamp=False):
        """
        Saves start and end times for training estimators.
        
        Args:
            timestamps_df (DataFrame): Dataframe containing: [strategy_name, 
                                                    begin_timestamp, 
                                                    end_timestamp,
                                                    difference between start and end timestamp]

            dataset_name(string): name of dataset on which the estimator was trained.
            overwrite_timestamp(Boolean): overwrite timestamps if they exist already
        """
        store = pd.HDFStore(self._hdf5_filename, self._mode)
        dts_path = f'{RUNTIMES_GROUP}/{dataset_name}'
        timestamp_exits = self.check_h5_path_exists(dts_path)
        if overwrite_timestamp is True or timestamp_exits is False:
            store[dts_path] = timestamps_df
        else:
            prior_timestamp, meta = self.load_dataset_pd(dts_path, return_metadata=False)
            timestamps_df = timestamps_df.append(prior_timestamp)
            store[dts_path] = timestamps_df
        store.close()
    
  
    def list_datasets(self, hdf5_group):
        """
        Lists all datasets/sub-groups in an HDF5 group

        Args:
            hdf5_group(string): hdf5_group: path to HDf5 group
        Returns:
            `array of strings`
        """
        datasets = []
        f = h5py.File(self._hdf5_filename, self._mode)
        if hdf5_group not in f:
            raise ValueError(f'Group {hdf5_group} does not exist in {self._hdf5_filename,}')
        for i in f[hdf5_group].items():
            datasets.append(i[0])
        f.close()
        return datasets

    def load_dataset_h5(self, dataset_path):
        """
        Loads dataset from HDF5 database. The dataset needs to have been saved as a numpy array originally

        :type dataset_path: string
        :param dataset_path: path to dataset.

        :rtype: `numpy array, metadata dictionary`
        """
        f = h5py.File(self._hdf5_filename, self._mode)
        idx = f[dataset_path][...]
        #load metadata
        meta = f[dataset_path].attrs.items()
        meta_dict = {}
        for m in meta:
            meta_dict[m[0]] = m[1]
        f.close()
        return idx, meta_dict
        
    def load_dataset_pd(self, dataset_path, return_metadata=True):
        """
        Loads dataset from HDF5 database. The dataset needs to have been saved as a pandas Datafram originally

        Args:
            dataset_path(string): path to dataset.
            return_metadata(Boolean): Flag whether the metadata should be returned
        Returns:
            `pandas DataFrame, metadata dictionary`
        """
        store = pd.HDFStore(self._hdf5_filename, self._mode)
        dataset = store[dataset_path]
        if return_metadata:
            metadata = store.get_storer(dataset_path).attrs.metadata
        else:
            metadata = None
        store.close()
        return dataset, metadata
    def save_pandas_dataset(self, dataset, save_loc, metadata, verbose=False):
        """
        Saves a dataset stored in DataFrame format in the database

        Parameters
        -----------
        dataset (DataFrame): Dataset that will be saved in the databse
        save_path(string): group in HDF5 database where the data will be saved
        metadata(JSON): Must contain ``class_name``, and ``dataset_name`` key-value pairs
        """

        store = pd.HDFStore(self._hdf5_filename, self._mode)
        dts_name = metadata['dataset_name']
        save_path = f'{save_loc}/{dts_name}'
        store[save_path] = dataset
        store.get_storer(save_path).attrs.metadata = metadata
        store.close()

    def save_datasets(self, datasets, datasets_save_paths, dts_metadata, verbose = False):
        #TODO This function needs to be removed together with its implementation in the data package
        '''
        saves datasets in HDF5 database. 
        dataset_names must contain full path.

        :type datasets: array of pandas DataFrame
        :param datasets: array of datasets formatted as pandas DataFrame.

        :type datasets_save_paths: array of string
        :param datasets_save_paths: Array with the save locations for each dataset.

        :type dts_meta: array of dictionaries
        :param dts_meta: Metadata for each dataset.

        :type verbose: boolean
        :param verbose: Display or not progress with saving the datasets.
        '''
        store = pd.HDFStore(self._hdf5_filename, self._mode)
        for dts in zip(datasets, dts_metadata, datasets_save_paths):
            
            dts_name = dts[1]['dataset_name']     
            save_loc = dts[2]    
            store[save_loc] = dts[0]
            store.get_storer(save_loc).attrs.metadata = dts[1]
            if verbose is True:
                print(f'Saved: {dts_name} to HDF5 database')
        store.close()
    
    def split_dataset(self, dataset_path, test_size=0.33):
        """
        Splits dataset in train and test set.

        :type dataset_path: string
        :param dataset_path: Path to dataset

        :type test_size: float
        :param test_size: Fraction of samples that will be put in test set

        :rtype: tuple of four arrays (X_train, X_test, y_train, y_test)
        """
        #load
        dataset, metadata = self.load_dataset(dataset_path)
        class_name = metadata['class_name']
        dataset_name = metadata['dataset_name']
        #split
        y = dataset[class_name]
        X = dataset.loc[:, dataset.columns != class_name]
        X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42)
        #reformat y_train and y_test
        return (X_train, X_test, y_train,  
                y_test)
         
 