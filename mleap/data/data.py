from ..shared.files_io import FilesIO
from sklearn.model_selection import train_test_split
from ..shared.static_variables import TRAIN_IDX, TEST_IDX, SPLIT_DTS_GROUP
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
        dts_list = hdf5_io.list_datasets(hdf5_group)
        dts_list = [hdf5_group + '/' + dts for dts in dts_list]
        return dts_list
    
    def open_hdf5(self, hdf5_path):
        return FilesIO(hdf5_path)

    def split_datasets(self, hdf5_in, hdf5_out, dataset_paths, split_datasets_group=None, test_size=0.33, random_state=1, verbose=False):
        
        if split_datasets_group is None:
            split_datasets_group = SPLIT_DTS_GROUP

        for dts_loc in dataset_paths:
            #split
            dts, metadata = hdf5_in.load_dataset(dts_loc)

            idx_dts_rows = dts.shape[0]
            idx_split = np.arange(idx_dts_rows)
            train_idx, test_idx =  train_test_split(idx_split, test_size=test_size, random_state=random_state)
            train_idx = np.array(train_idx)
            test_idx = np.array(test_idx)
            #save
            class_name = metadata['class_name']
            dataset_name = metadata['dataset_name']
            save_split_dataset_paths = [
                split_datasets_group + '/' + dataset_name,
                split_datasets_group + '/' + dataset_name 
            ]

            meta = [{'dataset_name': dataset_name}]*2
            names = ['train', 'test']

            if verbose is True:
                print(f'Saving split for: {dataset_name}')
            hdf5_out.save_array_hdf5(datasets=[train_idx, test_idx],
                                   group=split_datasets_group + '/' + dataset_name,
                                   array_names=names,
                                   array_meta=meta)
            
   