# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 15:35:16 2020

@author: rw1816
"""
import numpy as np
import pandas as pd
import os,sys
import math

## init
my_root_dir = "F:\\OneDrive - Imperial College London\\Post-doc\\build_data\\b1_220920\\"
hdw_root_dir = "F:\\code\\HDW_HS_analysis\\Results"

# read in my data
L1142_stats_df = pd.read_hdf(os.path.join(my_root_dir, 'data.h5'),  key='dfL1142')

#read in harry data
harry_df_global = pd.read_hdf(os.path.join(hdw_root_dir, 'FullFocusData.h5'), key='dfGlobalRes')

#%% cylinder breaks?
cylinderBreaks = [0]
footageBreaks = [0]
buildBreaks = [0]

for i in range(0,len(harry_df_global.loc[:,'Frame Number'])-1):
    diff = harry_df_global.loc[i+1,'Frame Number'] - harry_df_global.loc[i,'Frame Number']
    if diff > 1:
        #print('Cylinder')
        cylinderBreaks.append(i)
    elif diff < 1:
        cylinderBreaks.append(i)
        if harry_df_global.loc[i+1,'Build'] == harry_df_global.loc[i,'Build']:
            #print('Footage')
            footageBreaks.append(i)
        else:
            #print('Build')
            buildBreaks.append(i)
cylinderBreaks.append(len(harry_df_global.loc[:,'Frame Number'])-1)

#%% Train-Test-Validate Split

"""
-----------Do I even need all this stuff? - RW--------------------------
"""
cats = ['Spot area', 'Spot major axis', 'Spot minor axis','Spot max intensity',
       'Spot mean intensity', 'Spatter number', 'Spatter total area','Spatter mean area', 'Spatter median area', 'Spatter sample skewness',
       'Spatter mean max intensity', 'Spatter median max intensity',
       'Spatter intensity skewness']
dream_team = ['Spot area', 'Spot major axis','Spatter number', 'Spot minor axis','Spot max intensity','Spatter mean max intensity','Spatter mean area','Spatter sample skewness']
temp = []

frmPerSig = 2500

TrackDf = pd.DataFrame(columns=dream_team+['Porosity','Build']) #creates a new dataframe that's empty

# for each FOOTAGE
counter = 0
for i in range(1,len(cylinderBreaks)):
    entry = cylinderBreaks[i-1]
    exit = cylinderBreaks[i]
    temp = [i*frmPerSig for i in range(0,math.ceil((exit-entry)/frmPerSig))]
    
    # for each TRACK
    for j in range(1,len(temp)):
        TrackDf.loc[counter] = np.sum(harry_df_global.loc[entry+temp[j-1]:entry+temp[j],harry_df_global.columns.isin(list(cats))],axis=0)
        TrackDf.loc[counter,'Porosity'] = 100-harry_df_global.loc[exit,'Porosity'] 
        TrackDf.loc[counter,'Build'] = harry_df_global.loc[exit,'Build']

        counter =  counter+1
        
# Remove values without Porosity Vals
TrackDf = TrackDf[TrackDf.Porosity != 100]
# Fill in Nans as 0
TrackDf.fillna(0)

TrainTest_df = TrackDf.loc[TrackDf['Build']!=TrackDf['Build'].unique()[1]]
Validate_df = TrackDf.loc[TrackDf['Build']==TrackDf['Build'].unique()[1]]

#%% KNN
""" ------seems to me this cell is doing everything----------"""
 
import sklearn.neighbors
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

standard_scaler = preprocessing.MinMaxScaler()
X = standard_scaler.fit_transform(TrackDf[dream_team])
y = TrackDf['Porosity']
X_train, X_test, y_train_2, y_test_2 = train_test_split(X, y, test_size=.2,random_state=0)

""" ------- this is a small training set, total 790 rows split 80-20% train test. Why
do you pull out a single track for training? would it not be best to train on the full
c. 2m lines master df ??? -----------------------------------------------------"""


p=0.6
y_train = y_train_2>p
y_test = y_test_2>p
parameters = [{'n_neighbors': [5,6], 'weights': ['distance'],'p':[2]}]
knn = sklearn.neighbors.KNeighborsClassifier()
clf = sklearn.model_selection.GridSearchCV(knn,parameters,scoring= 'roc_auc',cv=10)
clf.fit(X_train, y_train)
y_score =  clf.predict_proba(X_test)

pd.concat([pd.Series(list(y_test_2)),pd.Series(y_score[:,1])], axis=1).to_csv('KNN_Porosity_4POD.csv')

#%%
df=L1142_stats_df.fillna(0)
mytest = df[dream_team]
myX=standard_scaler.fit_transform(mytest)
my_av=np.mean(myX, 0)
result=clf.predict(my_av.reshape(1, -1))
print('is the porosity worse than {0}%? .... {1}'.format(p, result[0]))
