import os
import urllib.request
from glob import glob
from scipy.io import arff
import pandas as pd
import tarfile

DELGADO_DIR = 'Delgado_data/'
DELGADO_DATASET_DOWNLOAD_URL='https://persoal.citius.usc.es/manuel.fernandez.delgado/papers/jmlr/data.tar.gz'
USE_PROXY = True 

class DownloadAndConvertDelgadoDatasets(object):
    

    def download_and_extract_datasets(self, verbose = None):
        #download
        filename = DELGADO_DATASET_DOWNLOAD_URL.split('/')[-1]
        if not os.path.isfile(DELGADO_DIR + filename):
            if verbose is True:
                print('Downloading Delgado dataset...')
            if USE_PROXY == True:
                proxy  = urllib.request.ProxyHandler({'https': '127.0.0.1:3128'} )
                opener = urllib.request.build_opener(proxy)
                urllib.request.install_opener(opener)
            urllib.request.urlretrieve(DELGADO_DATASET_DOWNLOAD_URL, DELGADO_DIR + filename)
        #extract
        delgado_dataset_dirs = glob(DELGADO_DIR+'*')
        if len(delgado_dataset_dirs) < 120:
            if verbose is True:
                print('Extracting datasets...')
            tar = tarfile.open(DELGADO_DIR + filename, "r:gz")
            tar.extractall(DELGADO_DIR)
            tar.close()

        #reformat datasets
        if verbose is True:
            print('Reformatting datasets...')
        dataset_dirs = glob(DELGADO_DIR+'*')
        
        datasets = []
        metadata = []
        for i in range(len(dataset_dirs)):
            if os.path.isdir(dataset_dirs[i]) is True:
                ds_dir = dataset_dirs[i]
                arff_files_in_dir = glob(ds_dir + os.sep +'*.arff')
                if len(arff_files_in_dir) == 2:
                    filename = arff_files_in_dir[0].split(os.sep)[-1]
                    filename = filename.split('.')[0]
                    filename = filename.split('_')[0] #test/train _ split
                    filename = filename.replace('-','_')
                    dts_name = filename.split('.')[0]
                    data1 = arff.loadarff(arff_files_in_dir[0])
                    data2 = arff.loadarff(arff_files_in_dir[1])
                    df1 = pd.DataFrame(data1[0])
                    df2 = pd.DataFrame(data2[0])
                    result = df1.append(df2, ignore_index=True)
                    result['clase'] = pd.to_numeric(result['clase'])

                elif len(arff_files_in_dir) == 1:
                    filename = arff_files_in_dir[0].split(os.sep)[-1]
                    filename = filename.split('.')[0]
                    filename = filename.replace('-','_')
                    dts_name = filename.split('.')[0]
                    data = arff.loadarff(arff_files_in_dir[0])
                    result = pd.DataFrame(data[0])
                    result['clase'] = pd.to_numeric(result['clase'])

                else:
                    print('Error: Dataset {0} has a different number of arff files'.format(dataset_dirs[i]))
                if verbose is True:
                    print(f'Loading: {dts_name}...')
                #return three arrays with data, name and metadata
                datasets.append(result)
                metadata.append({'class_name':'clase', 
                                'source':'Delgado',
                                'dataset_name': dts_name
                })
        return datasets, metadata
        