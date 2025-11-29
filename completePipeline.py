from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import make_column_selector

#import pandas as pd
import numpy as np

from columnTransformer import ColumnTransformer, cat_pipeline
from customTransformerKMEANS import ClusterSimilarity
"""the pipeline will do and why:
Missing values in numerical features will be imputed by replacing them with the median, as most ML algorithms don’t expect missing values. In categorical features, missing values will be replaced by the most frequent category.
The categorical feature will be one-hot encoded, as most ML algorithms only accept numerical inputs.
A few ratio features will be computed and added: bedrooms_ratio, rooms_​per_house, and people_per_house. Hopefully these will better correlate with the median house value, and thereby help the ML models.
A few cluster similarity features will also be added. These will likely be more useful to the model than latitude and longitude.
Features with a long tail will be replaced by their logarithm, as most models prefer features with roughly uniform or Gaussian distributions.
All numerical features will be standardized, as most ML algorithms prefer when all features have roughly the same scale."""

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ration_name(function_transformer, feature_names_in):
    return ['ratio']

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy='median'),
        FunctionTransformer(column_ratio, feature_names_out=ration_name),
        StandardScaler()
    )

log_pipeline = make_pipeline(
    SimpleImputer(strategy='median'),
    FunctionTransformer(np.log, feature_names_out='one-to-one'),
    StandardScaler())

cluster_simil = ClusterSimilarity(n_clusters=10, gamma=1.0, random_state=42)
default_num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())

preprocessing = ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline, ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_simil, ["latitude", "longitude"]),
        ("cat", cat_pipeline, make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline)
