# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 17:58:49 2022

@author: thepw
"""



#RUN LINE 16-85 FIRST 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from sklearn.datasets import make_blobs
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt
from numpy import where
import pandas as pd 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.layers import Dropout
from tensorflow.keras import models
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import max_norm
import datetime
from dask import dataframe as dd
from imblearn import under_sampling, over_sampling
from keras.models import model_from_json

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_epochs = 50
batch_size = 250
neurons_first=[500, 250]
neurons_second=[250, 200]
neurons_third=[200, 150]
neurons_fourth=[150, 100]
neurons_fifth=[100, 75]
neurons_sixth=[75, 50]
neurons_seventh=[50, 25]



"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import csv
from pandas import read_csv
filename = "Minority_Prediction_Train_Test_Set.csv"

for skip_blank_lines in [True, False]:
    print("Skip Blank Lines:", skip_blank_lines)
    df = read_csv(filename, skip_blank_lines=skip_blank_lines)

    print("Row count:", len(df))
    print("Unique values:", df[df.columns[0]].unique(), "\n")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print(df.shape)

print(df.shape)



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

#X.shape
#print(X.info)


# Use StandardScaler
Scale = StandardScaler()
X = pd.DataFrame(Scale.fit_transform(X), columns = X.columns)

# Convert X into a numpy array for keras/tensorflow 
X = np.asarray(X)

# Split data into training/test sets 
trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.80, shuffle= True, random_state = 25)


# Balance classes - over sample from the minority class (1)
ros = RandomOverSampler(random_state=42)
trainX, trainY = ros.fit_resample(trainX, trainY)

# Convert Y into categorical variables 
trainY=to_categorical(trainY)
testY=to_categorical(testY)


# Confirm training/test dataset shapes 
print(trainX.shape)
print(testX.shape)
print(trainY.shape)
print(testY.shape)

# Set seed to reproduce results 
np.random.seed(2532)

input_dim = trainX.shape[1] #Establish input dimension parameter
n_classes = 2 #Establish output size parameter


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Final Model
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

activation_func = 'relu'
start_time = datetime.datetime.now()
model = Sequential()
model.add(Dense(512, input_dim=trainX.shape[1], activation=activation_func))
model.add(Dense(512,activation =activation_func, kernel_initializer='normal'))
model.add(Dense(256,activation =activation_func, kernel_initializer='normal'))
model.add(Dense(128,activation =activation_func, kernel_initializer='normal'))
model.add(Dense(64,activation =activation_func, kernel_initializer='normal'))
model.add(Dense(32,activation =activation_func, kernel_initializer='normal'))#try a smaller layer, maybe 200
model.add(Dense(n_classes, activation='sigmoid'))
opt = SGD(learning_rate=0.2, momentum=0.7)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(trainX, trainY, epochs = num_epochs, batch_size = batch_size, validation_data=(testX, testY), verbose = 1)
print(model.evaluate(testX, testY))

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
    
    

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
 Print out results for final model 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Create a chart to display training and test accuracy by epoch   
plt.figure(0)
# Pull training accuracy history 
plt.plot(history.history['accuracy'],'r')
# Pull test accuracy history
plt.plot(history.history['val_accuracy'],'g')
# Assign number of ticks on the x-axis
plt.xticks(np.arange(0, num_epochs+5, 2.0))
# Define dimensions of the chart 
plt.rcParams['figure.figsize'] = (8, 6)
# Define label for x-axis 
plt.xlabel("Num of Epochs")
# Define label for y-axis 
plt.ylabel("Accuracy")
# Define title for chart
plt.title("Training Accuracy vs Validation Accuracy")
# Define label for legend
plt.legend(['train','validation'])
 

# Create a chart to display training and test loss rates by epoch 
plt.figure(1)
# Pull training loss rate history
plt.plot(history.history['loss'],'r')
# Pull test loss rate history 
plt.plot(history.history['val_loss'],'g')
# Assign number of ticks on the x-axis 
plt.xticks(np.arange(0, num_epochs+5, 2.0))
# Define dimensions of the chart
plt.rcParams['figure.figsize'] = (8, 6)
# Define label for x-axis 
plt.xlabel("Num of Epochs")
# Define label for y-axis 
plt.ylabel("Loss")
# Define title for chart
plt.title("Training Loss vs Validation Loss")
# Define label for legend
plt.legend(['train','validation'])
#Show plots
plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                  horizontalalignment="center",
                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



# Predict the values from the validation dataset
pred_label = model.predict(testX)
# Convert predictions classes to one hot vectors 
pred_label_classes = np.argmax(pred_label,axis = 1) 
# Convert validation observations to one hot vectors
label_true = np.argmax(testY,axis =1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(label_true, pred_label_classes) 
print(confusion_mtx)
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(testY.shape[1])) 

stop_time = datetime.datetime.now()
print ("Time required for training:",stop_time - start_time)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Test Optimal nodes and Layers: Version 1 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# start_time = datetime.datetime.now()
# # Function to create model, required for KerasClassifier
# def create_model(neurons_first=1, neurons_second=2, neurons_third=3,neurons_fourth=4,neurons_fifth=5,neurons_sixth=6,neurons_seventh=7):
#     # create model
#     model = Sequential()
#     model.add(Dense(neurons_first, input_dim=trainX.shape[1], kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(neurons_second, kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(neurons_third, kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(neurons_fourth, kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(neurons_fifth, kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(neurons_sixth, kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(neurons_seventh, kernel_initializer='normal', activation='relu', kernel_constraint=max_norm(4)))
#     model.add(Dense(n_classes, activation='sigmoid'))
#     optimizer = 'adam'
#     # Compile model
#     model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
#     return model
# 	
# # create model
# model = KerasClassifier(build_fn=create_model, epochs=num_epochs, batch_size=batch_size, verbose=1)
# # define the grid search parameters
# # neurons_first = [5, 6]
# # neurons_second = [8, 9]
# param_grid = dict(neurons_first=neurons_first, neurons_second=neurons_second,neurons_third=neurons_third,
#                   neurons_fourth=neurons_fourth,neurons_fifth=neurons_fifth,neurons_sixth=neurons_sixth,
#                   neurons_seventh=neurons_seventh)
# #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# # n_jobs=-1 means using all processors (don't use).

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=1)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# grid_result = grid.fit(trainX, trainY)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# print out results
# print("The best accuracy is: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("Mean = %f StDev = %f with: %r" % (mean, stdev, param))

 

# stop_time = datetime.datetime.now()
# print ("Time required for training:",stop_time - start_time)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TEST NODES AND LAYERS: Version 2 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Accuracy = 0 #Establish initial MSE value
# best_nodes = 0
# best_layers = 0

# start_time = datetime.datetime.now()
# # fit model with given number of nodes, returns test set accuracy
# def evaluate_model(n_nodes, n_layers, trainX, trainy, testX, testy):
#     # configure the model based on the data
#     n_input, n_classes = trainX.shape[1], testY.shape[1]
#     # define model
#     model = Sequential()
#     for _ in range(1, n_layers):
#         model.add(Dense(n_nodes, input_dim=n_input, activation='relu', kernel_initializer='normal'))
#     model.add(Dense(n_classes, activation='softmax'))
#     # compile model
#     opt = SGD(learning_rate=0.01, momentum=0.5)
#     model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#     # fit model on train set
#     history = model.fit(trainX, trainY, epochs=num_epochs, batch_size = batch_size, verbose=1)
#     # evaluate model on test set
#     _, test_acc = model.evaluate(testX, testY, verbose=1)
#     return history, test_acc


# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Show output Section
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # evaluate model and plot learning curve with given number of nodes

# for n_layers in num_layers:
#     for n_nodes in num_nodes:
#         # evaluate model with a given number of nodes
#         history, result = evaluate_model(n_nodes, n_layers, trainX, trainY, testX, testY)
#         # summarize final test set accuracy
#         print('nodes=%d, layers=%d: %.3f' % (n_nodes, n_layers, result))
#         if result >= Accuracy:
#             Accuracy = result
#             best_node = n_nodes
#             best_layers = n_layers
#         # plot learning curve
#         plt.plot(history.history['loss'], label=str(n_nodes )+ 'n, ' + str(n_layers)+'l')
    
# #add confusion matrix for best combo of nodes and layers        
# print('Best nodes=%d, Best layers=%d: Best accuracy: %.3f' % (best_node, best_layers, Accuracy))
# # show the plot
# plt.legend.loc='center right'
# plt.legend()
# plt.show()
# stop_time = datetime.datetime.now()
# print ("Time required for training:",stop_time - start_time)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
TEST LEARNING RATE AND MOMENTUM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# start_time = datetime.datetime.now()

# # Function to create model, required for KerasClassifier
# def create_model(learn_rate=0.01, momentum=0.01):
#     # create model
#     activation_func = 'relu'
#     model = Sequential()
#     model.add(Dense(2000, input_dim=trainX.shape[1], activation=activation_func))
#     model.add(Dense(2000,activation =activation_func, kernel_initializer='normal'))
#     model.add(Dense(2000,activation =activation_func, kernel_initializer='normal'))
#     model.add(Dense(2000,activation =activation_func, kernel_initializer='normal'))
#     model.add(Dense(2000,activation =activation_func, kernel_initializer='normal'))
#     model.add(Dense(n_classes, activation='sigmoid'))
#     opt = SGD(learning_rate=0.3, momentum=0.8)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
 	
# # create model
# model = KerasClassifier(build_fn=create_model, epochs=num_epochs, batch_size=batch_size, verbose=1)
# # define the grid search parameters
# learn_rate = [0.2, 0.3]
# momentum = [.8,.10]
# param_grid = dict(learn_rate=learn_rate, momentum=momentum)
# #https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# # n_jobs=-1 means using all processors (don't use).

# grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, verbose=1)
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Train Model Section
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# grid_result = grid.fit(trainX, trainY)

# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Show output Section
# """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# # print out results
# print("The best accuracys is: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

# stop_time = datetime.datetime.now()
# print ("Time required for training:",stop_time - start_time)
