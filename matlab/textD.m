
%TEXTD �˴���ʾ�йش˺�����ժҪ
%tf ����ĳ�����������������г��ֵ�Ƶ��
%dif ����ĳ�������������������еĳ���Ƶ��
%ti_dif�������ĵ÷�
%data����ÿ�������г��ֵĺ������Ƶ��
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

     