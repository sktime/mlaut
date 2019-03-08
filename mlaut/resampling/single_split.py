from mlaut.resampling.mlaut_resampling import MLaut_resampling
from sklearn.model_selection import train_test_split

class Single_Split(MLaut_resampling):
    """
    Wrapper for sklearn.model_selection.train_test_split
    
    The constructor implements the same parameters as sklearn.model_selection.train_test_split
    
    Parameters
    ----------
    test_size : float, int or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
    
    """
    def __init__(self,test_size=0.25, train_size=None, random_state=None, shuffle=True, stratify=None):
        self._test_size=test_size
        self._train_size=train_size
        self._random_state=random_state
        self._shuffle=shuffle
        self._stratify=stratify
    def resample(self, data):
        pass

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
                names = [self._train_idx, self._test_idx]

                if verbose is True:
                    logging.info(f'Saving split for: {dataset_name}')
                self._output_h5_file.save_array_hdf5(datasets=[train_idx, test_idx],
                                    group=path_to_save,
                                    array_names=names,
                                    array_meta=meta)
            split_dts_list.append(self._split_datasets_group + '/' + dataset_name)

            self._split_dts_list = split_dts_list
        
        return split_dts_lis