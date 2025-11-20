from zlib import crc32
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from housing_data import housing_full
#Random Sampling
"""#if you run the program again, it will generate a different test set
def shuffle_and_split_data(data, test_ratio, rng):
    shuffle_indices = rng. permutation(len(data))
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffle_indices[:test_set_size]
    train_indices = shuffle_indices[test_set_size:] 

    return data.iloc[train_indices], data.iloc[test_indices]

rng = np.random.default_rng()
train_set, test_set = shuffle_and_split_data(housing_full, 0.2, rng)
print(len(train_set))
print(len(test_set))
"""
# adds an index
housing_with_id = housing_full.reset_index()

# Random Samping
# Will ensure same test set; However REQUIRES id_column in dataset
def is_id_in_test_set(identifer, test_ratio):
    return crc32(np.int64(identifer)) < test_ratio * 2**32

def split_data_with_id_hash(data, test_ratio, id_column):
    ids= data[id_column]
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, 'index')

#stratified sampling 

#create five income cateories 
housing_full["income_cat"] = pd.cut(housing_full['median_income'],
                                    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                    labels=[1, 2, 3, 4, 5])

"""
# Visual of the 5 cateories 
cat_counts = housing_full['income_cat'].value_counts().sort_index()
cat_counts.plot.bar(rot=0, grid=True)
plt.ylabel('Number of distrcits')
plt.xlabel('Income category')
plt.show()
"""