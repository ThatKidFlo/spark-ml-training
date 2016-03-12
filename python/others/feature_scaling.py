import pandas as pd
import numpy as np
from sklearn import preprocessing

df = pd.io.parsers.read_csv(
    'https://raw.githubusercontent.com/rasbt/pattern_classification/master/data/wine_data.csv',
     header=None,
     usecols=[0,1,2],
    nrows = 20
    )

df.columns=['Class label', 'Alcohol', 'Malic acid']

df.head()


std_scale = preprocessing.StandardScaler().fit(df[['Alcohol']])
df_std = std_scale.transform(df[['Alcohol']])

minmax_scale = preprocessing.MinMaxScaler().fit(df[['Alcohol']])
df_minmax = minmax_scale.transform(df[['Alcohol']])

print(df['Alcohol'].values)
print(df_std)

scaled_values = [p[0] for p in df_std]
print scaled_values

total = sum([p*p for p in scaled_values])
print total