# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('veriler.csv')

print(veriler)

boy= veriler[['boy']]

print(boy)

boykilo = veriler [['boy','kilo']]

print(boykilo)