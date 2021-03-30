%���ݴ���

%content��������
data1=data(1:250,:)
size=size(data,1)
data2=data(size-249:size,:)
data=[data1;data2]


%���з���
a=cell(20,1)
for i=1:20
    j=A1(1,i)
    k=A(j,1)
    a(i,:)=cellstr(a1(k,:))
end




%������ת��ΪFASTA��ʽ
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
%��txt�ļ�����Ϊ�ɲ�������
%��������
%��Ӭ
data1=data1(1:6022,:)
data2=data2(1:6000,:)

l1=ones(6022,1)
l2=zeros(6000,1)
l3=[l1;l2]

data1=data(1:6022,:)
data2=data(17135+1:17135+6000,:)
data=[data1;data2]


%����
data1=data1(1:2332,:)
data2=data2(1:2331,:)
l1=ones(2332,1)
l2=zeros(2331,1)
l3=[l1;l2]


data1=data(1:2332,:)
data2=data(5506+1:5506+2331,:)
data=[data1;data2]


%С��

l1=ones(2380,1)
l2=zeros(2380,1)
l3=[l1;l2]

data=seq
data1=data(1:2380,:)
data2=data(7307+1:7307+2380,:)
data=[data1;data2]
 %�ϲ�����
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
    




%���ݼ���
n = randperm(size(data,1));
%%
data= data(n(1:2000),:);
l3 = l3(n(2001:2500),:);

data=data(1:2000,:)

data= data(1:3000,:);
l3 = l3(1:3000,:);

%��ȡ����������

data=bend(seq);

%����f-score
load('pseknc2.2.mat')
data1=[l3 data]
%��ƽ�ĸ��
    pos=textread('S.c.positive.txt','%s')
    neg=textread('S.c.negative.txt','%s')
    
%2����������
for i=1:1:length(pos)/2
    pos(i,:)=[];
end
for i=1:1:length(neg)/2
   neg(i,:)=[]; 
end
pos=char(pos)
neg=char(neg)

%���е�����
sequence=[pos ;neg]
data=feature01(sequence)
%4���������ݱ�ǩ
l1=ones(405,1)
l2=zeros(406,1)
l3=[l1;l2]

%��ȡƵ������
load('seq.mat')

load('allsequence2.mat')
data=pinci2(sequence,allsequence)

load('allsequence3.mat')
data=pinci3(sequence,allsequence)

load('allsequence4.mat')
data=pinci4(sequence,allsequence)

load('allsequence5.mat')
data=pinci5(sequence,allsequence)

%��ȡGC����
data=GC(sequence)

%���Ͻ�
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

%ѵ�����Ͳ��Լ���ǩ
l1=ones(500,1)
l2=zeros(500,1)

%����������ǩ
l1=ones(1515,1)
l2=zeros(1515,1)
l3=[l1;l2]

l1=ones(6350,1)
l2=zeros(6351,1)
l3=[l1;l2]

%�ϲ�֢����������seq
%������1515��
%������1515��

%��ȡ�����Ʊ�������
load('seq.mat')
 data=feature01(data)
%��ȡƵ������
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

%��ȡGC����
data=GC(seq)

%��Ӭ

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

%����������ǩ
l1=ones(17135,1)
l2=zeros(17115,1)
l3=[l1;l2]


%��ȡ�����Ʊ�������
load('seq.mat')
data=feature01(seq)
%��ȡƵ������
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

%��ȡGC����
data=GC(seq)

%����
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

%��Сд��ĸ��Ϊ��д
sequence=data
s1=size(sequence,1);      %��С��������ĸ�������������
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


%����������ǩ
l1=ones(5506,1)
l2=zeros(5513,1)
l3=[l1;l2]





%��ȡ����������
 
load('allsequence2.mat')
data=pinci2(data,allsequence)

load('allsequence3.mat')
data=pinci3(data,allsequence)

load('allsequence4.mat')
data=pinci4(data,allsequence)

load('allsequence5.mat')
data=pinci5(data,allsequence)

%��ȡGC����
data=GC(seq)

%�����³
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

%��ȡ�����Ʊ�������
load('seq.mat')
data=feature01(seq)
%��ȡƵ������
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

%��ȡGC����
data=GC(seq)

%С��
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

%��ȡ����������
 
load('allsequence2.mat')
data=pinci2(seq,allsequence)

load('allsequence3.mat')
data=pinci3(seq,allsequence)

load('allsequence4.mat')
data=pinci4(seq,allsequence)

load('allsequence5.mat')
data=pinci5(seq,allsequence)

%��ȡGC����
data=GC(seq)

%�ϳ��ĸ
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

%��ֳ��ĸ
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

%���������������
allsequence=allSequence(6)
allsequence=char(allsequence)


%�������У������������е�Сд��Ϊ��д
data=trainSet
s1=size(data,1);  %����
s2=size(data,2);%����
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

