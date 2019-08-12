from sklearn.utils import resample
import numpy as np
import pandas as pd

class Whitebox_Bootstrap:
    """
    Wrapper for whitebox estimator modelling. Implements the bootstrap procedure

    Parameters
    ----------
    estimator : object
        Estimator that implements the `coefficients()` method
    bootstrap_runs : int
        Number of bootstrap runs to be performed
    """
    def __init__(self, estimator, bootstrap_runs=5):
        self._estimator = estimator
        self._bootstrap_runs = bootstrap_runs
        self._coefficients = dict()
    
    def fit(self, metadata, data, coefficient_names, random_state=1):
        for n in range(self._bootstrap_runs):
            #resample with replacement
            res_idx = resample(np.arange(data.shape[0]), random_state=random_state)
            est_fitted = self._estimator.fit(metadata=metadata, data=data.iloc[res_idx])
            fitted_estimator_coeffs = self._estimator.coefficients(fitted_estimator=est_fitted, coefficinet_names=coefficient_names)

            for k in fitted_estimator_coeffs.keys():
                if k in self._coefficients:
                    self._coefficients[k].append(fitted_estimator_coeffs[k])
                else:
                    self._coefficients[k] = [fitted_estimator_coeffs[k]]
    
    def coefficients(self):
        mean_coeffs  = []
        ste_coeffs = []

        for k in self._coefficients.keys():
            mean_coeffs.append(np.mean(self._coefficients[k]))
            ste_coeffs.append(np.std(self._coefficients[k]) / np.sqrt(len(self._coefficients[k])) )
        
        result_df = pd.DataFrame({'COEFFICIENT':mean_coeffs,
                                  'STE': ste_coeffs}, index=self._coefficients.keys())
        return result_df

        