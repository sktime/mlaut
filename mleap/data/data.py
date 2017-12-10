from ..shared.files_io import FilesIO
from sklearn.model_selection import train_test_split
from ..shared.static_variables import TRAIN_IDX, TEST_IDX, SPLIT_DTS_GROUP, EXPERIMENTS_PREDICTIONS_DIR
import numpy as np
class Data(object):

    def pandas_to_db(self, save_loc_hdf5, datasets, dts_metadata, save_loc_hdd):
        save_paths = []
        for dts in dts_metadata:
            save_paths.append(save_loc_hdf5 + dts['dataset_name'])
        files_io = FilesIO(save_loc_hdd)
        files_io.save_datasets(datasets=datasets, 
                               datasets_save_paths=save_paths, 
                               dts_metadata=dts_metadata)
        return files_io
    
    def list_datasets(self, hdf5_group, hdf5_io):
        dts_names_list = hdf5_io.list_datasets(hdf5_group)
        dts_names_list_full_path = [hdf5_group  + dts for dts in dts_names_list]
        return dts_names_list, dts_names_list_full_path
    
    def open_hdf5(self, hdf5_path, mode='a'):
        return FilesIO(hdf5_path, mode)

    def split_datasets(self, hdf5_in, hdf5_out, dataset_paths, split_datasets_group=None, test_size=0.33, random_state=1, verbose=False):
        split_dts_list = []
        if split_datasets_group is None:
            split_datasets_group = SPLIT_DTS_GROUP

        for dts_loc in dataset_paths:
            #split
            dts, metadata = hdf5_in.load_dataset_pd(dts_loc)
            idx_dts_rows = dts.shape[0]
            idx_split = np.arange(idx_dts_rows)
            train_idx, test_idx =  train_test_split(idx_split, test_size=test_size, random_state=random_state)
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            #save
            dataset_name = metadata['dataset_name']
            meta = [{u'dataset_name': dataset_name}]*2
            names = [TRAIN_IDX, TEST_IDX]

            if verbose is True:
                print(f'Saving split for: {dataset_name}')
            hdf5_out.save_array_hdf5(datasets=[train_idx, test_idx],
                                   group=split_datasets_group + '/' + dataset_name,
                                   array_names=names,
                                   array_meta=meta)
            split_dts_list.append(split_datasets_group + '/' + dataset_name)
        return split_dts_list
    
    def load_train_test_split(self, hdf5_out, dataset_name, 
                              split_dts_group=SPLIT_DTS_GROUP, 
                              train_idx=TRAIN_IDX, 
                              test_idx=TEST_IDX):
        train, train_meta = hdf5_out.load_dataset_h5(split_dts_group +'/'+ \
        dataset_name + '/' + train_idx)
        
        test, test_meta = hdf5_out.load_dataset_h5(split_dts_group +'/'+ \
        dataset_name + '/' + test_idx)
        
        return train, test, train_meta, test_meta

    def load_predictions(self, hdf5_out, dataset_name, 
                         experiments_predictions_dir=EXPERIMENTS_PREDICTIONS_DIR):
        pass

    def load_true_labels(self, hdf5_in, dataset_loc, lables_idx):
        dataset, meta = hdf5_in.load_dataset_pd(dataset_loc)
        labels_col_name = meta['class_name']
        return dataset[labels_col_name].iloc[lables_idx]
        


            
   