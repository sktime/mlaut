from src.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mleap import Data

delgado = DownloadAndConvertDelgadoDatasets()
datasets, dataset_names, metadata = delgado.download_and_extract_datasets(verbose = True)

data = Data()
data.pandas_to_db('delgado_datasets/', datasets, metadata, 'data/delgado.hdf5')