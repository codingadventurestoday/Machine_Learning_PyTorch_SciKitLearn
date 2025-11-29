from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

import pandas as pd

from stratified_test_set import housing
from completePipeline import preprocessing

housing_labels = housing["median_house_value"].copy()

tree_reg = make_pipeline(preprocessing, DecisionTreeRegressor())
tree_reg.fit(housing, housing_labels)

housing_predictions = tree_reg.predict(housing)
tree_rmse = root_mean_squared_error(housing_labels, housing_predictions)

tree_rmses = -cross_val_score(tree_reg, housing, housing_labels, 
                              scoring='neg_root_mean_squared_error', cv=10)

print(pd.Series(tree_rmses).describe())