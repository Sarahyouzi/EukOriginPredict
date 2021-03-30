# -*- coding: utf-8 -*-
"""
Created on Mon Jan 2 16:33:36 2019

@author: wanru
"""
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
import scipy
from numpy import *
import pandas as pd
from pandas import read_csv
from matplotlib import pyplot
from pandas import set_option
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import random
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import scipy.io as scio
from sklearn.model_selection import KFold

#不适合归一化


dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\H.s人类\\pseknc2.2-21.mat"  #4,4
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\res-290.mat" #8,18
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\test.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\res-290.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\res-290.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\res-290.mat"

dataFile="D:\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\res-290.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\res-290.mat"


dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\pseknc2.2-35.mat"#'max_depth': 4, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\content2,4-2000.mat"#'max_depth': 2, 'min_samples_split': 2
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\res-184.mat"#'max_depth': 3, 'min_samples_split': 2
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\test.mat"#'max_depth': 3, 'min_samples_split': 2
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\res-184.mat" #8,18
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\res-184.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\res-184.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\res-184.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\res-184.mat"






dataFile="D:\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\pseknc2.23456_147.mat"#'max_depth': 4, 'min_samples_split': 28
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\content_2,4-2000.mat"#'max_depth': 5, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\res-165.mat"#'max_depth': 6, 'min_samples_split': 18
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\pseknc2.123456-147.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\pseknc2.123456-147.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\pseknc2.123456-147.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\pseknc2.23456-147.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\pseknc2.123456-147.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\pseknc2.1234-147.mat"





dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\pseknc2.2-20.mat"#'max_depth': 5, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\content2,4-2000.mat"#'max_depth': 2, 'min_samples_split': 2
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\res-264.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\test.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\res-264.mat" #8,18
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\res-264.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\res-264.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\res-264.mat"
dataFile="D:\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\res-264.mat"












dataFile="D:\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\pseknc2.4_130.mat"#max_depth': 5, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\content2,4.mat"#'max_depth': 2, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\res-89.mat"#'max_depth': 8, 'min_samples_split': 2

dataFile="D:\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\pseknc2.6_238.mat"#'max_depth': 9, 'min_samples_split': 28

dataFile="D:\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\pseknc2.4_174.mat"#'max_depth': 8, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\content_2,4.mat"#'max_depth': 2, 'min_samples_split': 2
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\res-171.mat"#'max_depth': 8, 'min_samples_split': 8

data=scio.loadmat(dataFile)
X=data['data']
X=data['res']

lableFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\lable-2000.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\H.s人类\\lable-test.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\lable-2000.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\lable-test.mat"

lableFile="D:\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\lable-2000.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\lable-2000.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\lable-test.mat"

lableFile="D:\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\lable.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\lable.mat"
lableFile="D:\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\lable.mat"

lable=scio.loadmat(lableFile)
Y=lable['l3']





num_folds=5
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)


scoring = 'accuracy'
#设置参数矩阵：
param_grid = {}
param_grid['min_samples_split'] = np.arange(2,30,2)
param_grid['max_depth'] = np.arange(2,10)
model=DecisionTreeClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
cv_results = zip(grid_result.cv_results_['mean_test_score'],
                 grid_result.cv_results_['std_test_score'],
                 grid_result.cv_results_['params'])
for mean, std, param in cv_results:
    print('%f (%f) with %r' % (mean, std, param))




r1=np.arange(0,10,1)
r2=np.arange(0,5,1)
c=mat(ones((10,4)))
res=mat(ones((10,1)))
for i in r1:
     # x：数据
    # y：类别标签
    # pct：训练集所占比例
    length = len(Y)
    rr=np.arange(0,length,1)
    index = round(length/5)
    indexes =np.random.choice(rr,length)
    data_X=X[indexes[0:length]]
    data_Y=Y[indexes[0:length]]
    n1=0
    n0=0
    n10=0
    n01=0
    for j in r2:

        if j==0 :
            X_test=data_X[0:index]
            Y_test=data_Y[0:index]
            X_train=data_X[index-1:length]
            Y_train=data_Y[index-1:length]
        elif j==4:
            X_test=data_X[index*j-1:index*(j+1)]
            Y_test=data_Y[index*j-1:index*(j+1)]
            X_train=data_X[0:index*(j+1)]
            Y_train=data_Y[0:index*(j+1)]
        else:
            #    random.shuffle(indexes)
            X_test=data_X[index*j-1:index*(j+1)-1]
            Y_test=data_Y[index*j-1:index*(j+1)-1]
            X_train1=data_X[0:index*j-1]
            Y_train1=data_Y[0:index*j-1]
            X_train2=data_X[index*(j+1)-1:length]
            Y_train2=data_Y[index*(j+1)-1:length]
            X_train=np.row_stack((X_train1,X_train2))
            Y_train=np.concatenate((Y_train1,Y_train2),axis=0)
        model =DecisionTreeClassifier(max_depth=8,min_samples_split=8)

        model.fit(X_train,Y_train)
        predicted=model.predict(X_test)
        matrix=confusion_matrix(Y_test,predicted)
        classes=['0','1']
        dataframe=pd.DataFrame(data=matrix,index=classes,columns=classes)
    #print(dataframe)
        da=dataframe.values
        n1=n1+da[1,0]+da[1,1]
        n0=n0+da[0,0]+da[0,1]
        n10=n10+da[1,0]
        n01=n01+da[0,1]
        Sn=1-n10/n1
        Sp=1-n01/n0
        Acc=1-(n10+n01)/(n1+n0)
        Mcc=(1-n10/n1-n01/n0)/((1+(n01-n10)/n1)*(1+(n10-n01)/n0)**0.5)
    #c[i,0]=Sn
    #c[i,1]=Sp
    #c[i,2]=Mcc
    c[i,3]=Acc*100

a=np.mean(c[:,0])
s=np.std(c[:,0], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)

a=np.mean(c[:,1])
s=np.std(c[:,1], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)

a=np.mean(c[:,2])
s=np.std(c[:,2], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)

a=np.mean(c[:,3])
s=np.std(c[:,3], ddof=1)
d=s/np.sqrt(10)
print(a)
print(d)


r1=np.arange(0,10,1)
res=mat(ones((10,1)))
for i in r1:
     # x：数据
    # y：类别标签
    # pct：训练集所占比例
   

#    random.shuffle(indexes)
    
    X_train = X
    Y_train = Y

    model =DecisionTreeClassifier()
    model.fit(X_train,Y_train)
    
    
    X_test = X
    Y_test = Y

    result=model.score(X_test,Y_test)
    res[i,0]=result*100
max=res.max(0)
min=res.min(0)
ave=(max+min)/2
value=max-ave
print(ave)
print(value)

