'''
Created on Oct 9, 2016

@author: ethan
''' 
#Testing neural network properties for plasma frequencies
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import shutil
import random
import os
import pandas as pd
from pymatgen.symmetry.groups import SymmetryGroup
import datetime
import random
import math

Columns = ['MPID','Formula','EnergyPerAtom','Volume','Density','NSites','SpaceGroup','Symmetry',
           'FormationEnergy','EHull','a','b','c','s','p','d','f','ENegDiff','ENegAvg','TotVol',
           'TotPolar','AvgPolar','DiffPolar','TotAffin','AvgAffin','DiffAffin','TotIonEn','AvgIonEn',
           'DiffIonEn','TotMass','AvgMass','DiffMass','RedMass','TotVDW','AvgVDW','DiffVDW','Gap(VASP)','IsMetal']
CategoryColumns=['Symmetry']
ContinuousColumns=['a','b','c','s','p','d','f','ENegDiff','ENegAvg','TotVol',
           'TotPolar','AvgPolar','DiffPolar','TotAffin','AvgAffin','DiffAffin','TotIonEn','AvgIonEn',
           'DiffIonEn','TotMass','AvgMass','DiffMass','RedMass','TotVDW','AvgVDW','DiffVDW']
'''ContinuousColumns=['EnergyPerAtom','Volume','Density','NSites',
           'FormationEnergy','EHull','a','b','c','s','p','d','f','ENegDiff','ENegAvg','TotVol',
           'TotPolar','AvgPolar','DiffPolar','TotAffin','AvgAffin','DiffAffin','TotIonEn','AvgIonEn',
           'DiffIonEn','TotMass','AvgMass','DiffMass','RedMass','TotVDW','AvgVDW','DiffVDW']'''
MetallicityColumn=['EHull']



def BuildRegressor(mod_dir): 
    symmetrygroup = tf.contrib.layers.sparse_column_with_hash_bucket('Symmetry', hash_bucket_size=1000)
    a = tf.contrib.layers.real_valued_column('a')
    b = tf.contrib.layers.real_valued_column('b')
    c = tf.contrib.layers.real_valued_column('c')
    s = tf.contrib.layers.real_valued_column('s')
    p = tf.contrib.layers.real_valued_column('p')
    d = tf.contrib.layers.real_valued_column('d')
    f = tf.contrib.layers.real_valued_column('f')
    enegdiff = tf.contrib.layers.real_valued_column('ENegDiff')
    eneg_buckets=tf.contrib.layers.bucketized_column(enegdiff, boundaries=[0.3,1.4])
    enegavg = tf.contrib.layers.real_valued_column('ENegAvg' )
    totv = tf.contrib.layers.real_valued_column('TotVol' )
    totpol = tf.contrib.layers.real_valued_column('TotPolar' )
    avgpol = tf.contrib.layers.real_valued_column('AvgPolar')
    diffpol = tf.contrib.layers.real_valued_column('DiffPolar')
    totaffin = tf.contrib.layers.real_valued_column('TotAffin' )
    avgaffin = tf.contrib.layers.real_valued_column('AvgAffin' )
    diffaffin = tf.contrib.layers.real_valued_column('DiffAffin' )
    totionen = tf.contrib.layers.real_valued_column('TotIonEn' )
    avgionen = tf.contrib.layers.real_valued_column('AvgIonEn' )
    diffionen = tf.contrib.layers.real_valued_column('DiffIonEn' )
    totmass = tf.contrib.layers.real_valued_column('TotMass' )
    avgmass = tf.contrib.layers.real_valued_column('AvgMass' )
    diffmass = tf.contrib.layers.real_valued_column('DiffMass' )
    redmass = tf.contrib.layers.real_valued_column('RedMass' )
    totvdw = tf.contrib.layers.real_valued_column('TotVDW' )
    avgvdw = tf.contrib.layers.real_valued_column('AvgVDW' )
    diffvdw = tf.contrib.layers.real_valued_column('DiffVDW' )
    
    #tf.contrib.layers.embedding_column(symmetrygroup, dimension=10),
    deep_columns=[tf.contrib.layers.embedding_column(symmetrygroup, dimension=10),
                  s,p,d,f,enegdiff,enegavg,totv,totpol,avgpol,diffpol,totaffin,avgaffin,
                  diffaffin,totionen,avgionen,diffionen,totmass,avgmass,diffmass,redmass,totvdw,avgvdw,diffvdw]
    wide_columns=[s,p,d,f,enegdiff,enegavg,totv,totpol,avgpol,diffpol,totaffin,avgaffin,
                  diffaffin,totionen,avgionen,diffionen,totmass,avgmass,diffmass,redmass,totvdw,avgvdw,diffvdw]

    
    model=tf.contrib.learn.DNNLinearCombinedRegressor(linear_feature_columns=wide_columns,
                                                      dnn_feature_columns=deep_columns,
                                                      dnn_hidden_units=network,
                                                      linear_optimizer=tf.train.FtrlOptimizer(learning_rate=0.1),
                                                      dnn_optimizer=tf.train.ProximalAdagradOptimizer(
                                                                                                  learning_rate=0.1,
                                                                                                  l1_regularization_strength=0.0,
                                                                                                  l2_regularization_strength=0.001),
                                                      dnn_dropout=0.2,
                                                      model_dir=mod_dir
                                                      )
    return model
 
def input_fn(df, train=False):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in ContinuousColumns}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {k: tf.SparseTensor(
      indices=[[i, 0] for i in range(df[k].size)],
      values=df[k].values,
      shape=[df[k].size, 1])
                        for k in CategoryColumns}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    if train:
        label = tf.constant(df[MetallicityColumn].values)
        # Returns the feature columns and the label.
        return feature_cols, label
    else:
        # so we can predict our results that don't exist in the csv
        return feature_cols

def train_and_eval():
    """Train and evaluate the model."""
    df_train = pd.read_csv(
        tf.gfile.Open("AllDescriptors.csv"),
        names=Columns,
        skipinitialspace=True,
        skiprows=1)
    #df_train.to_csv('ljk.csv')
    #print(df_train.dtypes)
    removelist=random.sample(range(len(df_train)), math.floor(len(df_train)/10)) #random list of points to remove
    keeplist=list(range(0, len(df_train)))
    keeplist=[a for a in keeplist if a not in removelist]
    df_train.drop(df_train.index[removelist], inplace=True) #actually remove
    #df_train.to_csv('ljkj.csv')
    df_test = pd.read_csv(
        tf.gfile.Open("AllDescriptors.csv"),
        names=Columns,
        skipinitialspace=True,
        skiprows=1)
    df_test.drop(df_test.index[keeplist], inplace=True)
    #df_test.to_csv('ljkj2.csv')

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
    m = BuildRegressor(model_dir)
    m.fit(input_fn=lambda: input_fn(df_train, True), steps=iterations)
    TrainingResults=m.predict(input_fn=lambda: input_fn(df_train))
    CrossValidationResults=m.predict(input_fn=lambda: input_fn(df_test))
    trainTotal=0
    trainCorrect=0
    valTotal=0
    valCorrect=0
    '''for i in range(0,len(TrainingResults)):
        #print(df_train['MPID'].values[i],df_train[MetallicityColumn].values[i],TrainingResults[i])
        trainTotal+=1
        if df_train[MetallicityColumn].values[i][0]==TrainingResults[i]:
            trainCorrect+=1
    trainAcc=trainCorrect/trainTotal
    for i in range(0,len(CrossValidationResults)):
        #print(df_train['MPID'].values[i],df_train[MetallicityColumn].values[i],TrainingResults[i])
        valTotal+=1
        if df_train[MetallicityColumn].values[i][0]==TrainingResults[i]:
            valCorrect+=1
    valAcc=valCorrect/valTotal
    f.write(str(trainAcc)+'\t'+str(valAcc)+'\n')'''
    
    
    
    #TestingResults=m.predict(input_fn=lambda: input_fn(df_test))
    #for i in range(0,len(TrainingResults)):
        #print(df_train[MetallicityColumn].values[i],TrainingResults[i])
    #    f1.write(str(df_train[MetallicityColumn].values[i][0])+'\t'+str(TrainingResults[i])+'\n')
    #results = m.evaluate(input_fn=lambda: input_fn(df_train, True), steps=1)
    TrainError=np.mean([abs(a-b) for a,b in zip(df_train[MetallicityColumn].values,TrainingResults)])
    TrainErrorVec.append(TrainError)
    #print(error,fracError)
    
    #for i in range(0,len(CrossValidationResults)):
    #    f2.write(str(df_test[MetallicityColumn].values[i][0])+'\t'+str(CrossValidationResults[i])+'\n')
    ValidError=np.mean([abs(a-b) for a,b in zip(df_test[MetallicityColumn].values,CrossValidationResults)])
    ValidErrorVec.append(ValidError)

    #for key in sorted(results):
    #    print("%s: %s" % (key, results[key]))
iterations=25000 #network optimization iterations
numIterations=20 #90-10 splits
ErrorVec=[]
fracErrorVec=[]
ev=[]
f=open('PredictEHull_1.txt','w')
#f=open('hold.txt','w')
f.write('#TrainCorrect'+'\t'+'CrossValCorrect'+'\n')
#f1=open('FullCrossValidation_Train.txt','w')
#f2=open('FullCrossValidation_Valid.txt','w')

for iter in range(0,numIterations):
    N=3
    #network=[5*N,4*N,3*N,2*N,1*N]
    network=[1*N,1*N]
    #network=[100]
    TrainErrorVec=[]
    ValidErrorVec=[]
    
    #network=[random.randint(5,30),random.randint(5,30),random.randint(5,30),random.randint(5,30),random.randint(5,30)]
    for trial in range(0,1):
        print('N = '+str(N)+', trial = '+str(trial)+', iteration = '+str(iter))
        model_dir = "/tmp/plasmonic_model"+str(datetime.date.today())+str(random.random())
        #print(model_dir)
        if os.path.exists(model_dir):
            shutil.rmtree(model_dir)
        train_and_eval()
        shutil.rmtree(model_dir)
    '''numLayers=len(network)
    fill=5-numLayers
    for i in range(0,len(network)):
        f.write(str(network[i])+'\t')
    for i in range(len(network),5):
        f.write(str(0)+'\t')'''
    #f.write(str(np.mean(fracErrorVec))+'\t'+str(np.mean(ErrorVec))+'\t'+str(sum(network))+'\n')
    print(np.mean(ValidErrorVec))
    f.write(str(np.mean(TrainErrorVec))+'\t'+str(np.mean(ValidErrorVec))+'\t'+str(0.1*np.mean(ValidErrorVec)+0.9*np.mean(TrainErrorVec))+'\n')
#for item in ev:
#    print(item)

f.close()
#f1.close()
#f2.close()
