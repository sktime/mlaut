from ..shared.files_io import FilesIO
from sklearn.model_selection import train_test_split
from ..shared.static_variables import (TRAIN_IDX, 
                                       TEST_IDX, 
                                       SPLIT_DTS_GROUP,
                                       EXPERIMENTS_PREDICTIONS_GROUP, 
                                       set_logging_defaults)
import numpy as np
import logging
class Data(object):
    """
    Interface class expanding the functionality of :func:`~mleap.shared.files_io.FilesIO`

    Parameters
    ----------
    hdf5_datasets_group: string
        HDF5 group of datasets in input HDF5 database file
    experiments_predictions_group: string
        localtion in HDF5 database where the predictions will be saved
    split_datasets_group: string
        location in HDF5 database were the test/train split will be saved.
    train_idx: string
        name of group where the indexes of the samples used for training are saved
    test_idx: string
        name of group where the indexes of the samples used for testing are saved
    verbose: boolean
        Display messages
    """
    def __init__(self, 
                hdf5_datasets_group,
                experiments_predictions_group=EXPERIMENTS_PREDICTIONS_GROUP,
                split_datasets_group=SPLIT_DTS_GROUP,
                train_idx=TRAIN_IDX,
                test_idx=TEST_IDX,
                verbose=True):
        self._experiments_predictions_group = experiments_predictions_group
        self._split_datasets_group=split_datasets_group
        self._train_idx_group=train_idx
        self._test_idx_group=test_idx
        self._hdf5_datasets_group = hdf5_datasets_group
        #contains the names of the datasets on which the experiments can be performed
        self._datasets=None
        self._verbose=verbose
        self._overwrite_resampling_splits=False
        
        #varibales for storing the data
        #HDF5 variables
        self._input_h5_file = None
        self._output_h5_file = None
        #in memory variables
        self._input_data = None
        self._metadata = None
        self._output_data = None

    def set_io(self, input_data, output_data, input_mode='a', output_mode='a'):
        """
        Setter function for a pointer to a HDF5 database. Wrapper for `self._open_hdf5()`

        Parameters
        ----------
            input_data: string
                path to HDF5 file saved on disk.
            input_mode: string
                open and create file modes as per the `h5py documentation <http://docs.h5py.org/en/latest/high/file.html>`_.
            output_data: string
                path to HDF5 file saved on disk.
            output_mode: string
                open and create file modes as per the `h5py documentation <http://docs.h5py.org/en/latest/high/file.html>`_.
        """
        self._input_h5_file = self._open_hdf5(input_data, input_mode)
        self._output_h5_file = self._open_hdf5(output_data, output_mode)
    def get_io(self):
        """
        Returns the handles to the input and output files

        Returns
        -------
        tuple:
            returns the handles to the input_io and the output_io files
        """

        return self._input_h5_file, self._output_h5_file
    def from_memory(self, input_data, metadata, output_data):
        """
        Provides functionality for running the experiments from memory

        Parameters
        ----------
        input_data: array of pandas DataFrame
            array of datasets
        metadata: array of dictionaries
            array of dictionaries with the metadata for each dataset
        output_data: mlaut output object
            instance of mlaut output object
        """
        self._input_data = input_data
        self._metadata = metadata
        self._output_data = output_data

    def get_datasets(self):
        """
        Returns the list of datasets available in the database on which the experiments can be performed.

        Returns
        -------
            array of string
        """
        #return self.list_datasets(self._hdf5_datasets_group)
        return self._datasets
    def set_datasets(self, dts_names):
        """
        Provides functionality for overriding the the default list of datasets on which the experiments will be performed.
        """
        self._datasets=dts_names
    def pandas_to_db(self, datasets, dts_metadata):
        """
        Saves array of datasets in pandas DataFrame format in HDf5 Database.
        This represents an interface method for :func:`~mleap.shared.files_io.FilesIO.save_datasets`

        Parameters
        ----------
            
        datasets: array of pandas DataFrame
            array of datasets formatted as pandas DataFrame.

        dts_meta: array of dictionaries
            Metadata for each dataset.
        """
        save_paths = []
        for dts in dts_metadata:
            save_paths.append(self._hdf5_datasets_group +'/'+ dts['dataset_name'])
        #files_io = FilesIO(save_loc_hdd)
        self._input_h5_file.save_datasets(datasets=datasets, 
                               datasets_save_paths=save_paths, 
                               dts_metadata=dts_metadata)
    
    def list_datasets(self, hdf5_group):
        """
        Returns sub group in parent HDF5 group.

        Parameters
        ----------
        hdf5_group: string
            Path to HDF5 parent group of which we are quering the subgroups.
        
        Returns
        -------
        tuple: 
            array with dataset names and array with full path to datasets.
        """
        dts_names_list = self._input_h5_file.list_datasets(hdf5_group)
        dts_names_list_full_path = [hdf5_group  +'/'+ dts for dts in dts_names_list]
        return dts_names_list, dts_names_list_full_path
    
    def _open_hdf5(self, hdf5_path, mode='a'):
        """
        Parameters
        ----------
            hdf5_path: string
                path to HDF5 file saved on disk.

            mode: string
                open and create file modes as per the `h5py documentation <http://docs.h5py.org/en/latest/high/file.html>`_.
        """
        return FilesIO(hdf5_path, mode)

    def load_predictions(self, dataset_name):
        """ 
        Loads the predictions for a particular dataset

        Parameters
        ----------
        dataset_name: string
            name of the dataset for which the predictions will be loaded

        Retuns
        -------
        list:
            Array in the form [[strategy name][predictions]]`.
        
        """

        load_path = f'{self._experiments_predictions_group}/{self._hdf5_datasets_group}/{dataset_name}'
        estimator_predictions = self._output_h5_file.list_datasets(load_path)
        predictions = []
        for estim in estimator_predictions:
            prediction_path = f'{load_path}/{estim}'
            prediction=self._output_h5_file.load_dataset_h5(prediction_path)
            predictions.append([estim,prediction])
        return predictions
        
    def save_resampling_splits(self, train_idx, test_idx,meta):
        """
        Saves the resampling splits in the database
        
        train_idx: numpy array
            numpy array with the indexes for the train data
        test_idx: numpy array
            numpy array with the indexes for the test data
        meta: dictionary
            dictionary with metadata for the dataset
        """
        dataset_name=meta['dataset_name']
        path_to_save = f'{self._split_datasets_group}/{self._hdf5_datasets_group}/{dataset_name}'

        if self._verbose is True:
            logging.info(f'Saving split for: {dataset_name}')
        
        split_exists = self._output_h5_file.check_h5_path_exists(path_to_save)
        if split_exists is True:
            if self._verbose is True:
                logging.warning(f'Skipping {dataset_name} as test/train split already exists in output h5 file.')
        
        if split_exists is not True or (split_exists and self._overwrite_resampling_splits): 
            self._output_h5_file.save_array_hdf5(datasets=[train_idx, test_idx],
                            group=path_to_save,
                            array_names=[self._train_idx_group, self._test_idx_group],
                            array_meta=[meta,meta])
        
    
    def split_datasets(self, 
                       test_size=0.33, 
                       random_state=1, 
                       verbose=True):
        """
        Splits datasets in test and train sets.

        Parameters
        ----------
            test_size: float
                percentage of samples to be put in the test set.

            random_state: integer
                random state for test/train split.

            verbose: boolean
                if True prints progress messages in terminal.

        Returns
        -------
            array of strings containing locations of split datasets.
        """
        split_dts_list = []
        
        if self._hdf5_datasets_group is None:
            raise ValueError('hdf5_datasets_group cannot be type None. Specify it in the constructor of the class.')


        _, dataset_paths = self.list_datasets(hdf5_group=self._hdf5_datasets_group)
        self._datasets = dataset_paths 
        for dts_loc in dataset_paths:
            #check if split exists in h5
            dts, metadata = self._input_h5_file.load_dataset_pd(dts_loc)
            dataset_name = metadata['dataset_name']
            path_to_save = f'{self._split_datasets_group}/{self._hdf5_datasets_group}/{dataset_name}'
            split_exists = self._output_h5_file.check_h5_path_exists(path_to_save)
            if split_exists is True:
                if verbose is True:
                    logging.warning(f'Skipping {dataset_name} as test/train split already exists in output h5 file.')
            else:  
                #split
                idx_dts_rows = dts.shape[0]
                idx_split = np.arange(idx_dts_rows)
                train_idx, test_idx =  train_test_split(idx_split, test_size=test_size, random_state=random_state)
                train_idx = np.array(train_idx)
                test_idx = np.array(test_idx)
                #save
                meta = [{u'dataset_name': dataset_name}]*2
                names = [self._train_idx_group, self._test_idx_group]

                if verbose is True:
                    logging.info(f'Saving split for: {dataset_name}')
                self._output_h5_file.save_array_hdf5(datasets=[train_idx, test_idx],
                                    group=path_to_save,
                                    array_names=names,
                                    array_meta=meta)
            split_dts_list.append(self._split_datasets_group + '/' + dataset_name)

            self._split_dts_list = split_dts_list
        
        return split_dts_list
    
    def load_train_test_split(self, dataset_name):
        """
        Loads test train split form HDF5 database.

        Parameters
        ----------
        dataset_name: string
            name of dataset for which the splits will be loaded.

        Returns
        -------
            tuple with train indices, test indices, train metadata and test metadata.
        """
        path_train = f'/{self._split_datasets_group}/{self._hdf5_datasets_group}/{dataset_name}/{self._train_idx_group}'

        train, train_meta = self._output_h5_file.load_dataset_h5(path_train, return_meta=True)
        path_test = f'/{self._split_datasets_group}/{self._hdf5_datasets_group}/{dataset_name}/{self._test_idx_group}'
        test, test_meta = self._output_h5_file.load_dataset_h5(path_test, return_meta=True)
        
        return train, test, train_meta, test_meta
    def load_non_resampled_dataset(self,dts_name):
        """
        Loads a dataset from the database

        Parameters
        ----------
        dts_name: string
            full path to the dataset
        
        Returns
        -------
            tuple with X, y, meta features, targets and metadata
        """
        if self._input_h5_file is not None:
            dts_df, meta = self._input_h5_file.load_dataset_pd(dts_name)
            label_column = meta['class_name']
        
        if self._input_data is not None:
            #TODO: algorithm for loading dataset from memory
            pass
        y = dts_df[label_column]
        X = dts_df.drop(label_column, axis=1)

        return X,y, meta

    def load_test_train_dts(self, dts_name):
        """
        Loads test/train data.

        Parameters
        ----------
        dts_name: string
            name of dataset for which the splits will be loaded.
        Returns
        -------
            tuple arrays in the form: X_train, X_test, y_train, y_test where X are the features and y are the lables.
        """
        train, test, _, _ = self.load_train_test_split(dts_name)
        dts, meta = self._input_h5_file.load_dataset_pd(f'{self._hdf5_datasets_group}/{dts_name}')
        label_column = meta['class_name']
        
        y_train = dts.iloc[train][label_column]
        y_train = np.array(y_train)
        
        y_test = dts.iloc[test][label_column]
        y_test = np.array(y_test)

        X_train = dts.iloc[train]
        X_train = X_train.drop(label_column, axis=1)
        X_train = np.array(X_train)
        
        X_test = dts.iloc[test]
        X_test = X_test.drop(label_column, axis=1)
        X_test = np.array(X_test)

        return X_train, X_test, y_train, y_test
        


    def load_true_labels(self, dts_name):
        """
        Loads labels for dataset

        Parameters
        ----------
        dts_name: string
            Name of dataset
        
        Returns
        -------
        pandas DataFrame
        """
        X_train, X_test, y_train, y_test = self.load_test_train_dts(dts_name)
        return y_test
        


            
   