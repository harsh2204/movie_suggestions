# import pandas as pd
import numpy as np

# df = pd.read_csv("jester-data-1.csv")
# print(df.head())

arr = np.genfromtxt('jester-data-1.csv',delimiter=',')
print(repr(arr))