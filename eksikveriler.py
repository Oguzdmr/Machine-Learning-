# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('eksikveriler.csv')

print(veriler)

boy= veriler[['boy']]

print(boy)

boykilo = veriler [['boy','kilo']]

print(boykilo)

from sklearn.impute import SimpleImputer
im = SimpleImputer(missing_values = np.nan, strategy = 'mean')

#from sklearn.preprocessing import SimpleImputer

#imputer=Imputer(missing_values='NaN',strategy='mean',axis=0)

Yas =veriler.iloc[:,1:4].values
print(Yas)
im.fit(Yas[:,1:4])
Yas[:,1:4] =im.transform(Yas[:,1:4])
print(Yas)