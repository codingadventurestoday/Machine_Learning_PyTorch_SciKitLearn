from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import root_mean_squared_error

from completePipeline import preprocessing
from stratified_test_set import housing

housing_labels = housing["median_house_value"].copy()

lin_reg = make_pipeline(preprocessing, LinearRegression())
lin_reg.fit(housing, housing_labels)

housing_predictions = lin_reg.predict(housing)

#print(housing_predictions[:5].round(-2))
#print(housing_labels.iloc[:5].values)

lin_rmse = root_mean_squared_error(housing_labels, housing_predictions)
print(lin_rmse)