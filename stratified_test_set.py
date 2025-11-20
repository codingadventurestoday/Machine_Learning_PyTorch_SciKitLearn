import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from housing_data import housing_full

housing_full["income_cat"] = pd.cut(housing_full['median_income'],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])
"""
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

strat_splits = []

for train_index, test_index in splitter.split(housing_full,
                                              housing_full['income_cat']):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]
"""

# shorter way to get stratified sampling 
strat_train_set, strat_test_set = train_test_split(housing_full, 
                                                   test_size=0.2, 
                                                   stratify=housing_full['income_cat'],
                                                   random_state=42)

print(f"strat_test_set: {strat_test_set['income_cat'].value_counts()}")

# drop income_cat as no longer needed
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()