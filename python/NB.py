# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:17:58 2019

@author: wanru
"""

from sklearn.naive_bayes import GaussianNB


#读取mat文件
import scipy.io as scio
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\pseknc2.4.mat"
data=scio.loadmat(dataFile)
X=data['data']


lableFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\lable.mat"
lable=scio.loadmat(lableFile)
Y=lable['l3']




num_folds=5
seed=7
kfold=KFold(n_splits=num_folds,random_state=seed)
#LDA数据将维
#lda = LinearDiscriminantAnalysis(n_components=2)
#lda.fit(X,Y)
#X = lda.transform(X)


model=GaussianNB()
result=cross_val_score(model,X,Y,cv=kfold)
print(result)
print(result.mean())


#正态化数据
from pandas import read_csv
from numpy import set_printoptions
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler().fit(X)
## 数据转换
X = scaler.transform(X)


r1=np.arange(0,10,1)
r2=np.arange(0,5,1)
c=mat(zeros((10,4)))
res=mat(zeros((10,1)))
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
        model =GaussianNB()
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
       # Sn=1-n10/n1
        #Sp=1-n01/n0
        Acc=1-(n10+n01)/(n1+n0)
        #Mcc=(1-n10/n1-n01/n0)/((1+(n01-n10)/n1)*(1+(n10-n01)/n0)**0.5)
    #c[i,0]=Sn
    #c[i,1]=Sp
    #c[i,2]=Mcc
    c[i,3]=Acc

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

