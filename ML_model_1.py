import pandas as pd

d = pd.read_csv("AllDescriptors.csv",header = 0, sep = ',')
print(list(d.columns.values))
print(d['MPID'].tolist() + ' ')
