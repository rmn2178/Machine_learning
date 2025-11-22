import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()

# ['DESCR', 'data', 'feature_names', 'frame', 'images', 'target', 'target_names']
print(dir(digits))