from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import pandas as pd

from stratified_test_set import housing 
"""We have a preprocessing pipeline that takes the entire training dataset and applies 
each transformer to the appropriate columns, then concatenates the transformed columns horizontally"""
num_pipeline = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())

num_attribs = ["longitude", "latitude", "housing_median_age", "total_rooms",
               "total_bedrooms", "population", "households", "median_income"]
cat_attribs = ["ocean_proximity"]

cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OneHotEncoder(handle_unknown='ignore'))

preprocessing = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

housing_prepared = preprocessing.fit_transform(housing)

df_housing_num_prepared = pd.DataFrame(
    housing_prepared, columns=num_pipeline.get_feature_names_out(),
    index=housing_prepared.index)