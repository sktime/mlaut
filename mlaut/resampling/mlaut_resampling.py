from abc import ABC, abstractmethod

class MLaut_resampling:
    """
    Abstact class that all MLaut resampling strategies should inherint from
    """
    @abstractmethod
    def resample(self):
        """
        
        """