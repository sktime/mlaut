from ..shared.files_io import FilesIO


class Data:

    def pandas_to_db(self, save_loc_hdf5, datasets, dts_metadata, save_loc_hdd):
        files_io = FilesIO(save_loc_hdd)
        files_io.save_datasets(save_loc_hdf5, datasets, dts_metadata)
    
    def list_datasets(self, hdf5_loc, hdf5_group):
        file_io = FilesIO(hdf5_loc)
        return file_io.list_datasets(hdf5_group)

    