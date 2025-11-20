import matplotlib.pyplot as plt

from housing_data import housing_full

"""housing_full.hist(bins=50, figsize=(12,8))
"""
housing_full["longitude"].hist(bins=50, figsize=(12,8))

plt.show()