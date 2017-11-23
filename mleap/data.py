from .files_io import FilesIO


class Data:

    def pandas_to_db(self, save_loc_hdf5, datasets, dts_metadata, save_loc_hdd):
        files_io = FilesIO(save_loc_hdd)
        files_io.save_datasets(save_loc_hdf5, datasets, dts_metadata)

    