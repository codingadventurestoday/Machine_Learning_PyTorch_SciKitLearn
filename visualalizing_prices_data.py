import matplotlib.pyplot as plt

from stratified_test_set import housing 

# s = district's population 
# c = color represents the price
# cmap = jet uses a predefine color map called jet
housing.plot(kind='scatter', x='longitude', y='latitude',
             grid=True, s=housing['population']/100, label='population',
             c='median_house_value', cmap='jet', colorbar=True,
             legend=True, sharex=False, figsize=(10, 7))

plt.show()