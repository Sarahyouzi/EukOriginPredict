load('pinci2.mat')
load('pinci3.mat')
load('pinci4.mat')
load('pinci5.mat')
load('pinci6.mat')
load('seq.mat')
load('mysequencenum.mat')
load('phy.mat')
r1=size(data,1);
%3����ȡ�ﻯ����
k=1
d=zeros(r1,30);
for n=1:r1
    len=300; 
    for a=1:5%r1�������������� 
        for e=1:6%r2��AGTC����������ϵ�����
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