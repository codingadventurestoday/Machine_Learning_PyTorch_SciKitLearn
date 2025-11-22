from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from stratified_test_set import housing 

housing_numerical = housing.select_dtypes(include=[np.number])

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
housing_num_min_scaled = min_max_scaler.fit_transform(housing_numerical)

std_scaler = StandardScaler()
housing_num_std_scaled = std_scaler.fit_transform(housing_numerical)
