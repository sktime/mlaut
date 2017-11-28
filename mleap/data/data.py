from ..shared.files_io import FilesIO


class Data:

    def pandas_to_db(self, save_loc_hdf5, datasets, dts_metadata, save_loc_hdd):
        save_paths = []
        for dts in dts_metadata:
            save_paths.append(save_loc_hdf5 + dts['dataset_name'])
        files_io = FilesIO(save_loc_hdd)
        files_io.save_datasets(datasets=datasets, 
                               datasets_save_paths=save_paths, 
                               dts_metadata=dts_metadata)
        return files_io
    
    def list_datasets(self, hdf5_loc, hdf5_group):
        file_io = FilesIO(hdf5_loc)
        dts_list = file_io.list_datasets(hdf5_group)
        dts_list = [hdf5_group + dts for dts in dts_list]
        return dts_list

    