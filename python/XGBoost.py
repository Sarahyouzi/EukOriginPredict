# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 10:04:28 2019

@author: wanru
"""

from xgboost import XGBClassifier

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\H.s人类\\pseknc2.2-21.mat"  #'learning_rate': 0.3, 'n_estimators': 10
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\H.s人类\\res-290.mat" #learning_rate': 0.1, 'n_estimators': 200

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 100
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 50
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\D.m果蝇\\pseknc2.4.mat"#learning_rate': 0.1, 'n_estimators': 200

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 200
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\pseknc2.4.mat"#'learning_rate': 0.4, 'n_estimators': 50
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\A.t拟南芥\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 100

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 50
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\pseknc2.4.mat"#'learning_rate': 0.5, 'n_estimators': 10
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\M.m小鼠\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 100

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\pseknc2.4.mat"#'learning_rate': 0.3, 'n_estimators': 100
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\pseknc2.4.mat"#'learning_rate': 0.1, 'n_estimators': 10
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\S.p裂殖酵母\\pseknc2.4.mat"#'learning_rate': 0.2, 'n_estimators': 100

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\P.p毕赤酵母\\pseknc2.4.mat"#'learning_rate': 0.5, 'n_estimators': 200

dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\pseknc2.4.mat"#'learning_rate': 0.3, 'n_estimators': 200
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\pseknc2.4.mat"#'learning_rate': 0.4, 'n_estimators': 10
dataFile="F:\\20210120\\IORI-PSEKNC2.0\\matlab\\K.l乳酸克鲁\\pseknc2.4.mat"#learning_rate': 0.4, 'n_estimators': 200





scoring = 'accuracy'
param_grid = {}
param_grid['n_estimators'] = [10,50, 100, 200, 300,400, 500]
param_grid['learning_rate'] = [0.1,0.2,0.3,0.4,0.5]
model=XGBClassifier()
kfold = KFold(n_splits=num_folds, random_state=seed)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result = grid.fit(X,Y)
print('最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_))
s='最优：%s 使用%s' % (grid_result.best_score_, grid_result.best_params_)
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
        j=0
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
            Y_train=np.row_stack((Y_train1,Y_train2))
        model =XGBClassifier(learning_rate=0.4,n_estimators=200)	
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
        #Sn=1-n10/n1
       # Sp=1-n01/n0
        Acc=1-(n10+n01)/(n1+n0)
       # Mcc=(1-n10/n1-n01/n0)/((1+(n01-n10)/n1)*(1+(n10-n01)/n0)**0.5)
   # c[i,0]=Sn
   # c[i,1]=Sp
   # c[i,2]=Mcc
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



