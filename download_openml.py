import openml
from mlaut.data import Data
import pandas as pd
from sklearn import preprocessing
import os
import sys

apikey = 'd2b1d13981d4abfb22895337baca924c'
openml.config.apikey = apikey
openml.config.set_cache_directory(os.path.expanduser('~/.openml/cache'))
NUMBER_OF_INSTANCES_CUTOFF_NUMBER = 10000 #

all_datasets = openml.datasets.list_datasets()

data = Data()
input_io = data.open_hdf5('data/openml.h5', mode='a')

for id in all_datasets.keys():
    #regression datasets have a value of -1. Classification datasets specify the number of classes
    if all_datasets[id]['NumberOfClasses'] == -1:
        print(f"Skipping dataset {id}, {all_datasets[id]['name']}. This is a regression dataset.")
        continue
    if all_datasets[id]['NumberOfMissingValues'] > 0:
        print(f"Skipping dataset {id}, {all_datasets[id]['name']} due to missing values.")
        continue

    if all_datasets[id]['NumberOfInstances'] > NUMBER_OF_INSTANCES_CUTOFF_NUMBER:
        print(f"Skipping dataset {id}, {all_datasets[id]['name']}. It has more than {NUMBER_OF_INSTANCES_CUTOFF_NUMBER} instances.")
        continue

    print(f"Trying to download dataset {id}, {all_datasets[id]['name']}")

    try:
        dataset = openml.datasets.get_dataset(id)
        X, names = dataset.get_data(return_attribute_names=True)
      

        metadata = {
            'class_name': dataset.__dict__['default_target_attribute'],
            'source': 'OpenML',
            'dataset_name':dataset.__dict__['name'],
            'dataset_id': id
        }
        class_name_index = names.index(metadata['class_name'])

        #Normalize the data
        # scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        # scaler.fit(X)
        # x_transformed  = scaler.transform(X)
        # x_transformed[:,class_name_index] = X[:, class_name_index]

        #Convert to DataFrame
        result = pd.DataFrame(X)
        result.columns=names
        result[metadata['class_name']] =  result[metadata['class_name']].astype(int)

        #save to hdf5
        input_io.save_pandas_dataset(dataset=result, save_loc='/openml', metadata=metadata)
        print(f"dataset {id}, {dataset.__dict__['name']} saved.")
    except KeyboardInterrupt:
        sys.exit()
    except:
        print(f"Cannot save dataset {id}, {all_datasets[id]['name']}")