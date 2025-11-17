# This is a code of loading iris flower dataset
import pandas as pd
from sklearn.datasets import load_iris

# Loading the data
iris = load_iris()

# The directories of the datasets
# 'DESCR', 'data', 'data_module', 'feature_names', 'filename', 'frame', 'target', 'target_names'
# The feature names of the datasets
# 'sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'

# Making the dataframe
df = pd.DataFrame(iris.data, columns=iris.feature_names)
#   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
#         5.1               3.5                1.4               0.2
#         4.9               3.0                1.4               0.2
#         4.7               3.2                1.3               0.2
#         4.6               3.1                1.5               0.2
#         5.0               3.6                1.4               0.2

# MAKING THE TARGET
# This adds a new column nae target
df['target'] = iris.target

# The target names of the datasets
# iris.target_names[0] = 'setosa'     = 0-49
# iris.target_names[1] = 'versicolor' = 50-99
# iris.target_names[2] = 'virginica'  = 100-149

# We are creating a new column named flower_name
df['flower_names'] = df.target.apply(lambda x: iris.target_names[x])

