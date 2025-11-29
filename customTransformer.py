from sklearn.preprocessing import FunctionTransformer
import numpy as np

from stratified_test_set import housing 

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing['population'])

print(log_pop)