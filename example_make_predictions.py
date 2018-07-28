from mlaut.data import Data
from mlaut.experiments import Orchestrator
from mlaut.estimators.estimators import instantiate_default_estimators
import os

data = Data()
input_io = data.open_hdf5('data/delgado.h5', mode='r')
out_io = data.open_hdf5('data/delgado-classification.h5', mode='a')

dts_names = os.listdir('data/trained_models')
orchest = Orchestrator(hdf5_input_io=input_io, 
                           hdf5_output_io=out_io,
                           dts_names=dts_names[0:3],
                           original_datasets_group_h5_path='/openml')

instantiated_models = instantiate_default_estimators(estimators=['Classification'], verbose=1)
orchest.predict_all(trained_models_dir='data/trained_models', estimators=instantiated_models, override=True)