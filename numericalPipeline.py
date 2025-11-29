from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd

from stratified_test_set import housing

num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('standardize', StandardScaler()),
])

housing_num = housing.select_dtypes(include=[np.number])

housing_num_prepared = num_pipeline.fit_transform(housing_num)
#print(housing_num_prepared[:2].round(2))

# recover a nice DataFrame
df_housing_num_prepared = pd.DataFrame(
    housing_num_prepared, clumns=num_pipeline.get_feature_names_out(),
    index=housing_num.index
)