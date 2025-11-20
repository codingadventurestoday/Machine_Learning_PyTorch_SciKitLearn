"""
1. Types of Machine Learning, Instance Based VS Model Based Learning  
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
"""
Turn model to KNearest neighbors regression
from sklearn.neighbors import KNeighborsRegressor
"""

# Download and prepare the data 
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values

#Visualize the data
lifesat.plot(kind="scatter", grid=True,
             x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Select a linear model 
model = LinearRegression()
"""
Turn model to KNearest neighbors regression
model = KNeighborsRegressor(n_neighbors=3)
"""
#train the model 
model.fit(X, y)

# make a prediction for Puerto Rico 
X_new = [[33_442.8]]
print(model.predict(X_new))