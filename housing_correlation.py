from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

from stratified_test_set import housing

corr_matrix = housing.corr(numeric_only=True)

#finds standard correlation coefficient of all attributes to median_house_value
corr_matrix['median_house_value'].sort_values(ascending=False)

attributes = ['median_house_value', 'median_income', 
              'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8))
#plt.show()

# zooming in on median_income correlation with median_house_value
housing.plot(kind='scatter', x='median_income', y='median_house_value',
             alpha=0.1, grid=True)

plt.show()