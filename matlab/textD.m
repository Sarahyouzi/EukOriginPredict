
%TEXTD 此处显示有关此函数的摘要
%tf 代表某个核苷酸在正样本中出现的频率
%dif 代表某个核苷酸在所有样本中的出现频率
%ti_dif代表最后的得分
%data代表每个序列中出现的核苷酸的频次
load('tf_idf.mat')
d=data
data1=[]
load('kmer1234-2.mat')
data1=data
    load('pinci1.mat')
    data1=[data1 data]
    load('pinci2.mat')
      data1=[data1 data]
    load('pinci3.mat')
      data1=[data1 data]
    load('pinci4.mat')
      data1=[data1 data]
    load('pinci5.mat')
      data1=[data1 data]
    load('pinci6.mat')
   data1=[data1 data]
  
   data=featureSelect(data,A,5460,350)
   d=d'
   d=featureSelect(d,A,5460,350)
   
   
   data=featureSelect(data,A,350,290)
   d=featureSelect(d,A,350,290)
   
    r1=size(data,1)
    r2=size(data,2)
    
    for i=1:r1
        for j=1:r2
        
            res(i,j)= data(i,j)*d(2,j)
        end
    end
    save('test.mat','res')

     