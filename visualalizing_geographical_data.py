import matplotlib.pyplot as plt


from stratified_test_set import housing

housing.plot(kind='scatter', x='longitude', y='latitude', grid=True, alpha= 0.2)

plt.show()