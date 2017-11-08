import os
from src.static_variables import DELGADO_DIR, DATA_DIR, EXPERIMENTS_TRAINED_MODELS_DIR

class CreateDirs:
    def create_directories(self):
        
        if not os.path.exists(DATA_DIR):
            print('Creating directory:{0}'.format(DATA_DIR))
            os.makedirs(DATA_DIR)

        if not os.path.exists(DELGADO_DIR):
            print('Creating directory:{0}'.format(DELGADO_DIR))
            os.makedirs(DELGADO_DIR)

        if not os.path.exists(EXPERIMENTS_TRAINED_MODELS_DIR):
            print('Creating directory:{0}'.format(EXPERIMENTS_TRAINED_MODELS_DIR))
            os.makedirs(EXPERIMENTS_TRAINED_MODELS_DIR)

