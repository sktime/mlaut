from mlaut.model_selection import single_split
import numpy as np

X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = single_split(X, y, test_size=0.33, random_state=42)

print(X_train)