# This is a code of loading iris flower dataset
# SUPPORT VECTOR MATH
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pickle

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

# Separates the data
df1 = df[df.target==0]
df2 = df[df.target==1]
df3 = df[df.target==2]

plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color='red', marker = '*')
plt.scatter(df2['petal length (cm)'], df2['petal width (cm)'],color='green', marker = '*')
plt.scatter(df3['petal length (cm)'], df3['petal width (cm)'],color='blue', marker = '*')
plt.xlabel('petal length (cm)')
plt.ylabel('petal width (cm)')

plt.show()

# Splitting the datasets
X = df.drop(['target','flower_names'], axis='columns')
Y = df.target
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Here the main model
model = SVC()
model.fit(X_train, Y_train)

# This code saves the model
with open('svm_model.pkl', 'wb') as file:
    pickle.dump(model, file)

with open('svm_model.pkl', 'rb') as file:
    model_x = pickle.load(file)

print(model_x.score(X_test, Y_test))