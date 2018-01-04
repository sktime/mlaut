from abc import ABC, abstractmethod
class MleapEstimator(ABC):

    @abstractmethod
    def build(self):
        """ Returns the estimator and its hyper parameters"""
    
    @abstractmethod
    def save(self):
        """ saves the trained model to disk """
    @abstractmethod
    def get_estimator_name(self):
        """ returs the name of the estimator"""    

    def set_trained_model(self, trained_model):
        self._trained_model = trained_model
    
    def get_trained_model(self):
        return self._trained_model