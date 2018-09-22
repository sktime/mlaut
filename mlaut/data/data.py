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

   
    :type split_datasets_group: string
    :param split_datasets_group: location in HDF5 database were the test/train split will be saved.

    :type train_idx: string
    :param train_idx: name of group where the indexes of the samples used for training are saved

    :type test_idx: string
    :param test_idx: name of group where the indexes of the samples used for testing are saved
    """
    def __init__(self, 
                experiments_predictions_group=EXPERIMENTS_PREDICTIONS_GROUP,
                split_datasets_group=SPLIT_DTS_GROUP,
                train_idx=TRAIN_IDX,
                test_idx=TEST_IDX):
        self._experiments_predictions_group = experiments_predictions_group
        self._split_datasets_group=split_datasets_group
        self._train_idx=train_idx
        self._test_idx=test_idx




    def pandas_to_db(self, save_loc_hdf5, datasets, dts_metadata, input_io):
        """
        Saves array of datasets in pandas DataFrame format in HDf5 Database.
        This represents an interface method for :func:`~mleap.shared.files_io.FilesIO.save_datasets`

        :type save_loc_hdf5: string
        :param save_loc_hdf5: Root group in HDF5 database where the datasets will be saved.

        :type datasets: array of pandas DataFrame
        :param datasets: array of datasets formatted as pandas DataFrame.

        :type dts_meta: array of dictionaries
        :param dts_meta: Metadata for each dataset.

        :type input_io: :func:`~mleap.shared.files_io.FilesIO`
        :param input_io: Instance of :func:`~mleap.shared.files_io.FilesIO` class.
        """
        save_paths = []
        for dts in dts_metadata:
            save_paths.append(save_loc_hdf5 + dts['dataset_name'])
        #files_io = FilesIO(save_loc_hdd)
        input_io.save_datasets(datasets=datasets, 
                               datasets_save_paths=save_paths, 
                               dts_metadata=dts_metadata)
    
    def list_datasets(self, hdf5_group, hdf5_io):
        """
        Returns sub group in parent HDF5 group.

        :type hdf5_group: string
        :param hdf5_group: Path to HDF5 parent group of which we are quering the subgroups.

        :type hdf5_io: :func:`~mleap.shared.files_io.FilesIO`
        :param hdf5_io: Instance of :func:`~mleap.shared.files_io.FilesIO`

        :rtype: tuple with array with dataset names and array with full path to datasets.
        """
        dts_names_list = hdf5_io.list_datasets(hdf5_group)
        dts_names_list_full_path = [hdf5_group  +'/'+ dts for dts in dts_names_list]
        return dts_names_list, dts_names_list_full_path
    
    def open_hdf5(self, hdf5_path, mode='a'):
        """
        :type hdf5_path: string
        :param hdf5_path: path to HDF5 file saved on disk.

        :type mode: string
        :param mode: open and create file modes as per the `h5py documentation <http://docs.h5py.org/en/latest/high/file.html>`_.
        """
        return FilesIO(hdf5_path, mode)

    def split_datasets(self, 
                       hdf5_in, 
                       hdf5_out, 
                       dataset_paths, 
                       test_size=0.33, 
                       random_state=1, 
                       verbose=True):
        """
        Splits datasets in test and train sets.

        :type hdf5_in: :func:`~mleap.shared.files_io.FilesIO` object
        :param hdf5_in: :func:`~mleap.shared.files_io.FilesIO` object where the original/input datasets are stored.

        :type hdf5_out: :func:`~mleap.shared.files_io.FilesIO` object
        :param hdf5_out: :func:`~mleap.shared.files_io.FilesIO` object where the split/output test/train indices are stored.

        :type dataset_paths: array of strings
        :param dataset_paths: full path to each dataset stored in the origial/input HDF5 database that will be split to test/train.

        :type test_size: float
        :param test_size: percentage of samples to be put in the test set.

        :type random_state: integer
        :param random_state: random state for test/train split.

        :type verbose: boolean
        :param verbose: if True prints progress messages in terminal.

        :rtype: array of strings containing locations of split datasets.
        """
        split_dts_list = []
        
        if not isinstance(dataset_paths, list):
            raise ValueError('dataset_paths must be an array')

        for dts_loc in dataset_paths:
            #check if split exists in h5
            dts, metadata = hdf5_in.load_dataset_pd(dts_loc)
            dataset_name = metadata['dataset_name']
            split_exists = hdf5_out.check_h5_path_exists(self._split_datasets_group + '/'+ dataset_name)
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
                names = [self._train_idx, self._test_idx]

                if verbose is True:
                    logging.info(f'Saving split for: {dataset_name}')
                hdf5_out.save_array_hdf5(datasets=[train_idx, test_idx],
                                    group=self._split_datasets_group + '/' + dataset_name,
                                    array_names=names,
                                    array_meta=meta)
            split_dts_list.append(self._split_datasets_group + '/' + dataset_name)
        
        return split_dts_list
    
    def load_train_test_split(self, hdf5_out, dataset_name):
        """
        Loads test train split form HDF5 database.

        :type hdf5_out: :func:`~mleap.shared.files_io.FilesIO` object
        :param hdf5_out: :func:`~mleap.shared.files_io.FilesIO` output object where the test/train index splits are stored.

        :type dataset_name: string
        :param dataset_name: name of dataset for which the splits will be loaded.

        :rtype: tuple with train indices, test indices, train metadata and test metadata.
        """
        path_train = f'/{self._split_datasets_group}/{dataset_name}/{self._train_idx}'
        train, train_meta = hdf5_out.load_dataset_h5(path_train)
        path_test = f'/{self._split_datasets_group}/{dataset_name}/{self._test_idx}'
        test, test_meta = hdf5_out.load_dataset_h5(path_test)
        
        return train, test, train_meta, test_meta

    def load_test_train_dts(self, hdf5_out, hdf5_in, dts_name, dts_grp_path):
        """
        Loads test/train data.

        :type hdf5_out: :func:`~mleap.shared.files_io.FilesIO` object
        :param hdf5_out: instance of :func:`~mleap.shared.files_io.FilesIO` object with output data.

        :type hdf5_in: :func:`~mleap.shared.files_io.FilesIO` object
        :param hdf5_in: instance of :func:`~mleap.shared.files_io.FilesIO` object with input data.

        :type dts_name: string
        :param dts_name: name of dataset for which the splits will be loaded.

        :type dts_grp_path: string
        :param dts_grp_path: path to root group where the original datasets are stored.

        :rtype: tuple arrays in the form: X_train, X_test, y_train, y_test where X are the features and y are the lables.
        """
        train, test, _, _ = self.load_train_test_split(hdf5_out,dts_name)
        dts, meta = hdf5_in.load_dataset_pd(f'{dts_grp_path}/{dts_name}')
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
        
    # def load_predictions(self, hdf5_out, dataset_name, 
    #                      experiments_predictions_dir=EXPERIMENTS_PREDICTIONS_DIR):
    #     pass

    def load_true_labels(self, hdf5_in, dataset_loc, lables_idx):
        """
        Loads labels for dataset

        :type hdf5_in: :func:`~mleap.shared.files_io.FilesIO` object
        :param hdf5_in: instance of :func:`~mleap.shared.files_io.FilesIO` object containing the original datasets

        :type lables_idx: string
        :param lables_idx: path the dataset. 

        :rtype: pandas DataFrame
        """
        dataset, meta = hdf5_in.load_dataset_pd(dataset_loc)
        labels_col_name = meta['class_name']
        return dataset[labels_col_name].iloc[lables_idx]
        


            
   