# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 03:36:00 2022

@author: thepw
"""
# Import required libraries 
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import numpy as np
from keras.models import model_from_json

# Read in file we want to populate with predictions 
df = pd.read_csv("Minority_Prediction_Blanks.csv", delimiter=',', header = 0)




# Fill NA's in forgivenessamount with median of column 
df['forgivenessamount'] = df['forgivenessamount'].fillna(df['forgivenessamount'].median())

# Define LabelEncoder 
L_Encode = LabelEncoder()

# Split df into X and Y variables 
Y = df['minority']
X = df.drop(['minority','loannumber', 'dateapproved', 'Size', 'Zip','Stalp'], axis=1)

# Reset index of X 
X = X.reset_index()

# Use label encoder to convert categorical values into numeric values 
X['NameFull'] = L_Encode.fit_transform(X['NameFull'])
X['City'] = L_Encode.fit_transform(X['City'])
X['Stcnty'] = L_Encode.fit_transform(X['Stcnty'])
X['CB'] = L_Encode.fit_transform(X['CB'])
X['Asset'] = L_Encode.fit_transform(X['Asset'])
X['HCAsset'] = L_Encode.fit_transform(X['HCAsset'])
X['borrowercity'] = L_Encode.fit_transform(X['borrowercity'])
X['borrowerstate'] = L_Encode.fit_transform(X['borrowerstate'])
X['originatinglender'] = L_Encode.fit_transform(X['originatinglender'])
X['originatinglendercity'] = L_Encode.fit_transform(X['originatinglendercity'])
X['originatinglenderstate'] = L_Encode.fit_transform(X['originatinglenderstate'])
X['naicscode'] = L_Encode.fit_transform(X['naicscode'])
X['ruralurbanindicator'] = L_Encode.fit_transform(X['ruralurbanindicator'])
X['lmiindicator'] = L_Encode.fit_transform(X['lmiindicator'])
X['FintechPartnership'] = L_Encode.fit_transform(X['FintechPartnership'])

#print(df.info)

# Use StandardScaler on X variables 
Scale = StandardScaler()
X = pd.DataFrame(Scale.fit_transform(X), columns = X.columns)

# Convert X into a numpy array for keras/tensorflow 
X = np.asarray(X)

# Load saved model in Json format 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
    
# Use loaded model on dataset with blank minority column 
df['minority'] = loaded_model.predict(X)  

# Convert prediction probabilities to 0, 1 
df['minority'] = np.where(df['minority']>= .5, '0', '1')

# Read back out as csv 
df.to_csv('Minority_Prediction_FilledIn.csv', index =False)
