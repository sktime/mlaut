import openml
from mlaut.data import Data
import pandas as pd
from sklearn import preprocessing
import os
import sys

apikey = 'd2b1d13981d4abfb22895337baca924c'
openml.config.apikey = apikey
openml.config.set_cache_directory(os.path.expanduser('~/.openml/cache'))

classification_tasks = openml.tasks.list_tasks(task_type_id=1)
regression_tasks = openml.tasks.list_tasks(task_type_id=2)

data = Data()
input_io = data.open_hdf5('data/openml.h5', mode='a')

for id in classification_tasks.keys():
    try:
        dataset = openml.datasets.get_dataset(id)
        X, names = dataset.get_data(return_attribute_names=True)

        #ignore datasets with empty values 
        num_missing_values = float(dataset.__dict__['qualities']['NumberOfMissingValues'])
        if num_missing_values > 0:
            print(f'skipping dataset {id}. Missing values')
            continue

        #ignore too big datasets
        if classification_tasks[id]['NumberOfInstances'] > 10000:
            print('skipping dataset {id}. It is too big')
            continue

        metadata = {
            'class_name': dataset.__dict__['default_target_attribute'],
            'source': 'OpenML',
            'dataset_name':dataset.__dict__['name']
        }
        class_name_index = names.index(metadata['class_name'])

        #Normalize the data
        scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
        scaler.fit(X)
        x_transformed  = scaler.transform(X)
        x_transformed[:,class_name_index] = X[:, class_name_index]

        #Convert to DataFrame
        result = pd.DataFrame(x_transformed)
        result.columns=names
        result[metadata['class_name']] =  result[metadata['class_name']].astype(int)

        #save to hdf5
        input_io.save_pandas_dataset(dataset=result, save_loc='/openml', metadata=metadata)
        print(f'dataset {id} saved.')
    except KeyboardInterrupt:
        sys.exit()
    except:
        print(f'Cannot save dataset {id}')