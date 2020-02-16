from sklearn.linear_model import LinearRegression
from mlaut.estimators.glm_estimators import Linear_Regression
import pydataset
from sklearn.datasets import load_boston

boston = pydataset.data('Boston')


meta = {
    'target': 'medv'
}
lr = Linear_Regression()
lr_fitted = lr.fit(metadata=meta, data=boston)

coeffs = lr.coefficients(lr_fitted, coefficinet_names=['crim', 'zn', 'indus','chas','nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio',
'black', 'lstat'])


from mlaut.estimators.whitebox import Whitebox_Bootstrap

wb = Whitebox_Bootstrap(estimator=Linear_Regression())

wb.fit(metadata=meta, data=boston, coefficient_names=['crim', 'zn', 'indus','chas','nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio',
'black', 'lstat'])

print(wb.coefficients())