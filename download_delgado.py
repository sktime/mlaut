from download_delgado.delgado_datasets import DownloadAndConvertDelgadoDatasets
from mlaut.data import Data

delgado = DownloadAndConvertDelgadoDatasets()

datasets, metadata = delgado.download_and_extract_datasets()
data = Data()
input_io = data.open_hdf5('data/delgado.h5', mode='a')
for item in zip(datasets, metadata):
    dts=item[0]
    meta=item[1]
    input_io.save_pandas_dataset(dataset=dts, save_loc='/openml', metadata=meta)
