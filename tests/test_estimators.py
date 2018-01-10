import unittest
import sys
sys.path.append('/media/viktor/Data/PhD/mleap')
from mleap.data.estimators import Logistic_Regression

# class TestEstimators(unittest.TestCase):
#     def test_inputs(self):
#         self.assertEqual( 2,2 )
#         #assert hasattr(Logistic_Regression, 'build') == True


# if __name__ == '__main__':
#     unittest.main()

def test_estimators():
    assert hasattr(Logistic_Regression, 'build') == True 