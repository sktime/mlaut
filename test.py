from sklearn.linear_model import LinearRegression
from mlaut.estimators.glm_estimators import Linear_Regression
import pydataset

longley = pydataset.data('longley')


meta = {
    'target': 'Employed'
}
lr = Linear_Regression()
lr_fitted = lr.fit(metadata=meta, data=longley)

coeffs = lr.coefficients(lr_fitted, coefficinet_names=['GNP.deflator', 'GNP', 'Unemployed',	'Armed.Forces',	'Population','Year'])


from mlaut.estimators.whitebox import Whitebox_Bootstrap

wb = Whitebox_Bootstrap(estimator=Linear_Regression())

wb.fit(metadata=meta, data=longley, coefficient_names=['GNP.deflator', 'GNP', 'Unemployed',	'Armed.Forces',	'Population','Year'])

print(wb.coefficients())