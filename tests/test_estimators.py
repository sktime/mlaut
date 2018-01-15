import unittest
import sys
sys.path.append('/media/viktor/Data/PhD/mleap')
from mleap.data.estimators import Lasso

def test_instantiation():
    Lasso.return_properties

def test_estimators():
    assert hasattr(Logistic_Regression, 'return_propertiess') == True 