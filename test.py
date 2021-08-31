from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.datasets import load_boston
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from Ridge_regression import Ridge_regression

boston = load_boston()
X, y = boston.data, boston.target
model = Ridge_regression(alpha=10)
model.fit(X, y)
# print(model.predit(X))
print(r2_score(y, model.predict(X)))

model2 = Ridge(alpha=10)
model2.fit(X, y)
print(r2_score(y, model2.predict(X)))