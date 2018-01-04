import abc 

class MleapEstimator(object):
    __metaclass__  = abc.ABCMeta
    @abc.abstractmethod
    def build(self):
        """ Returns the estimator and its hyper parameters"""