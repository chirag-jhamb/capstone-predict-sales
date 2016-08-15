#This file takes the downloaded data from kaggle and splits the data into training and testing sets so that it is ready for the project
#To be run only once for creation of files

import pandas as pd
import numpy as np
#read the downloaded dataset-
ds = pd.read_csv('download/store.csv') 
df = pd.read_csv('download/train.csv')
#split 85% of data into training set and rest into test set-
train = df.sample(frac=0.85,random_state=200)
test_n_results = df.drop(train.index)
#create test dataframe which will be used for prediction-
test=test_n_results.drop('Sales', 1)
#write everything into csv-
test.to_csv('inputs/test.csv')
test_n_results.to_csv('inputs/results.csv')
train.to_csv('inputs/train.csv')
ds.to_csv('inputs/store.csv')



