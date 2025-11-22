from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

from stratified_test_set import housing 

housing_cat = housing[['ocean_proximity']]
"""
ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# list of categories
#housing_cat_encoded.categories_
"""
# encodes categorical data with one hot method 
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

