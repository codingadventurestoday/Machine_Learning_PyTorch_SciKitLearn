from sklearn.impute import SimpleImputer
import numpy as np

from stratified_test_set import housing

strat_train_set = housing.copy()
#separate the predictors and the labels
housing = strat_train_set.drop('median_house_value', axis = 1)
housing_labels = strat_train_set['median_house_value'].copy()

# fit missing values in total_bedrooms
"""
# gets rid of corresponding districts
housing.dropna(subset=['total_bedrooms'], inplace=True)

# get rids of the whole attribute
housing.drop('total_bedrooms', axis=1, inplace=True)

# imputation: set missing values to median
median = housing['total_bedrooms'].median()
housing['total_bedrooms'] = housing['total_bedrooms'].fillna(median)
"""
imputer = SimpleImputer(strategy='median')
housing_numerical = housing.select_dtypes(include=[np.number])

imputer.fit(housing_numerical)
#print(imputer.statistics_)
#print(housing_numerical.median().values)

