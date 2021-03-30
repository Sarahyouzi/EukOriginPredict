%数据处理

%content特征处理
data1=data(1:250,:)
size=size(data,1)
data2=data(size-249:size,:)
data=[data1;data2]


%序列分析
a=cell(20,1)
for i=1:20
    j=A1(1,i)
    k=A(j,1)
    a(i,:)=cellstr(a1(k,:))
end




%将数据转化为FASTA格式
r1=size(data1)
r2=size(data1,2)
str1='>ID'
str3=' COMMENT'
fid=fopen('seq_fasta.txt','wt');
for i=1:r1
    fprintf(fid,'%s',str1);
     fprintf(fid,'%g',i);
     fprintf(fid,'%s\n',str3);
     str2=data1(i,:)
     fprintf(fid,'%s\n',str2);
end    
fclose(fid)
%将txt文件处理为可操作矩阵
%划分物种
%果蝇
data1=data1(1:6022,:)
data2=data2(1:6000,:)

l1=ones(6022,1)
l2=zeros(6000,1)
l3=[l1;l2]

data1=data(1:6022,:)
data2=data(17135+1:17135+6000,:)
data=[data1;data2]


%人类
data1=data1(1:2332,:)
data2=data2(1:2331,:)
l1=ones(2332,1)
l2=zeros(2331,1)
l3=[l1;l2]


data1=data(1:2332,:)
data2=data(5506+1:5506+2331,:)
data=[data1;data2]


%小鼠

l1=ones(2380,1)
l2=zeros(2380,1)
l3=[l1;l2]

data=seq
data1=data(1:2380,:)
data2=data(7307+1:7307+2380,:)
data=[data1;data2]
 %合并特征
 data1=[]

    load('pseknc2.1.mat')
    data1=[data1 data]
    load('pseknc2.2.mat')
      data1=[data1 data]
    load('pseknc2.3.mat')
      data1=[data1 data]
    load('pseknc2.4.mat')
      data1=[data1 data]
    load('pseknc2.5.mat')
      data1=[data1 data]
    load('pseknc2.6.mat')
   data1=[data1 data]
    




%数据减半
n = randperm(size(data,1));
%%
data= data(n(1:2000),:);
l3 = l3(n(2001:2500),:);

data=data(1:2000,:)

data= data(1:3000,:);
l3 = l3(1:3000,:);

%提取弯曲性特征

data=bend(seq);

%进行f-score
load('pseknc2.2.mat')
data1=[l3 data]
%酿酒酵母菌
    pos=textread('S.c.positive.txt','%s')
    neg=textread('S.c.negative.txt','%s')
    
%2、整理数据
for i=1:1:length(pos)/2
    pos(i,:)=[];
end
for i=1:1:length(neg)/2
   neg(i,:)=[]; 
end
pos=char(pos)
neg=char(neg)

%所有的序列
sequence=[pos ;neg]
data=feature01(sequence)
%4、样本数据标签
l1=ones(405,1)
l2=zeros(406,1)
l3=[l1;l2]

%提取频次特征
load('seq.mat')

load('allsequence2.mat')
data=pinci2(sequence,allsequence)

load('allsequence3.mat')
data=pinci3(sequence,allsequence)

load('allsequence4.mat')
data=pinci4(sequence,allsequence)

load('allsequence5.mat')
data=pinci5(sequence,allsequence)

%提取GC特征
data=GC(sequence)

%拟南芥
trainSet=textread('A.txt','%s')
testSet=textread('A_test.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
for i=1:1:length(testSet)/2
   testSet(i,:)=[]; 
end

trainSet=char(trainSet)
testSet=char(testSet)

%训练集和测试集标签
l1=ones(500,1)
l2=zeros(500,1)

%正负样本标签
l1=ones(1515,1)
l2=zeros(1515,1)
l3=[l1;l2]

l1=ones(6350,1)
l2=zeros(6351,1)
l3=[l1;l2]

%合并症负样本序列seq
%正样本1515条
%负样本1515条

%提取二进制编码特征
load('seq.mat')
 data=feature01(data)
%提取频次特征
load('seq.mat')

load('allsequence2.mat')
data=pinci2(data,allsequence)

load('allsequence3.mat')
data=pinci3(data,allsequence)

load('allsequence4.mat')
data=pinci4(data,allsequence)

load('allsequence5.mat')
data=pinci5(data,allsequence)


load('allsequence6.mat')
data=pinci6(seq,allsequence)

%提取GC特征
data=GC(seq)

%果蝇

pseq=[]

pseq=trainSet(1:3022,1:10)
pseq=[pseq;testSet(1:3000,1:10)]
pseq=[pseq;trainSet(1:2763,1:10)]
pseq=[pseq;testSet(1:2000,1:10)]

data1=trainSet(1:3350,:)
data1=[data1;testSet(1:3000,:)]


data2=trainSet(3351:6701,:)
data2=[data2;testSet(3001:6000,:)]


nseq=[]
data2(16000,:)
pseq(16000,:)

pseq=trainSet(3023:6022,1:10)
pseq=[pseq;testSet(3001:6000,1:10)]
pseq=[pseq;trainSet(2764:5527,1:10)]
pseq=[pseq;testSet(2001:4000,1:10)]
pseq=[pseq;trainSet(3351:6701,1:10)]
pseq=[pseq;testSet(3001:6000,1:10)]




trainSet=textread('S2.txt','%s')
testSet=textread('S2_test.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
for i=1:1:length(testSet)/2
   testSet(i,:)=[]; 
end

trainSet=char(trainSet)            
testSet=char(testSet)

data=feature01(seq)
data=feature01(testSet)

l1=ones(3000,1)
l2=zeros(3000,1)
l3=[l1;l2]

%正负样本标签
l1=ones(17135,1)
l2=zeros(17115,1)
l3=[l1;l2]


%提取二进制编码特征
load('seq.mat')
data=feature01(seq)
%提取频次特征
load('seq.mat')

load('allsequence2.mat')
data=pinci2(seq,allsequence)

load('allsequence3.mat')
data=pinci3(seq,allsequence)

load('allsequence4.mat')
data=pinci4(seq,allsequence)

load('allsequence5.mat')
data=pinci5(seq,allsequence)


load('allsequence6.mat')
data=pinci6(seq,allsequence)

%提取GC特征
data=GC(seq)

%人类
trainSet=textread('MCF7.txt','%s')
testSet=textread('MCF7_test.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
for i=1:1:length(testSet)/2
   testSet(i,:)=[]; 
end

trainSet=char(trainSet)
testSet=char(testSet)

data=feature01(seq)
data=feature01(testSet)

l1=ones(411,1)
l2=zeros(419,1)
l3=[l1;l2]

%将小写字母改为大写
sequence=data
s1=size(sequence,1);      %核小体的样本的个数（即列数）
for i=1:s1 
    for j=1:300
        if sequence(i,j)=='a'
            sequence(i,j)='A'
        elseif sequence(i,j)=='c'
            sequence(i,j)='C'
        elseif sequence(i,j)=='g'
            sequence(i,j)='G'
        elseif sequence(i,j)=='t'
            sequence(i,j)='T'
        end
    end
end


%正负样本标签
l1=ones(5506,1)
l2=zeros(5513,1)
l3=[l1;l2]





%提取二进制特征
 
load('allsequence2.mat')
data=pinci2(data,allsequence)

load('allsequence3.mat')
data=pinci3(data,allsequence)

load('allsequence4.mat')
data=pinci4(data,allsequence)

load('allsequence5.mat')
data=pinci5(data,allsequence)

%提取GC特征
data=GC(seq)

%乳酸克鲁
trainSet=textread('Kl.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
trainSet=char(trainSet)
sequence=trainSet
data=feature01(trainSet)
l1=ones(136,1)
l2=zeros(200,1)
l3=[l1;l2]

%提取二进制编码特征
load('seq.mat')
data=feature01(seq)
%提取频次特征
load('seq.mat')

load('allsequence2.mat')
data=pinci2(sequence,allsequence)

load('allsequence3.mat')
data=pinci3(sequence,allsequence)

load('allsequence4.mat')
data=pinci4(sequence,allsequence)

load('allsequence5.mat')
data=pinci5(sequence,allsequence)

load('allsequence6.mat')
data=pinci6(sequence,allsequence)

%提取GC特征
data=GC(seq)

%小鼠
trainSet=textread('P19.txt','%s')
testSet=textread('P19_test.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
for i=1:1:length(testSet)/2
   testSet(i,:)=[]; 
end

trainSet=char(trainSet)
testSet=char(testSet)

data=feature01(trainSet)
data=feature01(testSet)

l1=ones(7307,1)
l2=zeros(7307,1)
l3=[l1;l2]

%提取二进制特征
 
load('allsequence2.mat')
data=pinci2(seq,allsequence)

load('allsequence3.mat')
data=pinci3(seq,allsequence)

load('allsequence4.mat')
data=pinci4(seq,allsequence)

load('allsequence5.mat')
data=pinci5(seq,allsequence)

%提取GC特征
data=GC(seq)

%毕赤酵母
trainSet=textread('Pp.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
trainSet=char(trainSet)
sequence=trainSet
data=feature01(sequence)
l1=ones(268,1)
l2=zeros(300,1)
l3=[l1;l2]

load('allsequence2.mat')
data=pinci2(sequence,allsequence)

load('allsequence3.mat')
data=pinci3(sequence,allsequence)

load('allsequence4.mat')
data=pinci4(sequence,allsequence)

load('allsequence5.mat')
data=pinci5(sequence,allsequence)

%裂殖酵母
trainSet=textread('Sp.txt','%s')
for i=1:1:length(trainSet)/2
    trainSet(i,:)=[];
end
trainSet=char(trainSet)
sequence=trainSet
data=feature01(trainSet)
l1=ones(339,1)
l2=zeros(350,1)
l3=[l1;l2]

load('allsequence2.mat')
data=pinci2(sequence,allsequence)

load('allsequence3.mat')
data=pinci3(sequence,allsequence)

load('allsequence4.mat')
data=pinci4(sequence,allsequence)

load('allsequence5.mat')
data=pinci5(sequence,allsequence)

%所有序列排列组合
allsequence=allSequence(6)
allsequence=char(allsequence)


%处理序列，将序列中所有的小写换为大写
data=trainSet
s1=size(data,1);  %行数
s2=size(data,2);%列数
for i=1:s1 
    for j=1:s2
        if data(i,j)=='a'
            data(i,j)='A';
        end
        if data(i,j)=='c'
            data(i,j)='C';
        end
        if data(i,j)=='g'
            data(i,j)='G';
        end
        if data(i,j)=='t'
            data(i,j)='T';
        end
    end
end

