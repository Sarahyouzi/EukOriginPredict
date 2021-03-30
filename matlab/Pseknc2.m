load('pinci2.mat')
load('pinci3.mat')
load('pinci4.mat')
load('pinci5.mat')
load('pinci6.mat')
load('seq.mat')
load('mysequencenum.mat')
load('phy.mat')
r1=size(data,1);
%3、提取物化特征
k=1
d=zeros(r1,30);
for n=1:r1
    len=300; 
    for a=1:5%r1是样本集的行数 
        for e=1:6%r2是AGTC所有排列组合的行数
            t=0
            for i=1:(len-k-a)
               t=t+J(seq,n,e,i,a,mysequence,pyh);
            end
            t=t/(len-k-a);
            d(n,e+(a-1)*6)=t;
        end
    end
end

% selectedfeature=featureSelect(data,A,1024,883)
data=[data d]