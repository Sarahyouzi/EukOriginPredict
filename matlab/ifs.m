
function [record] = ifs(A,data,l3,cmd,num)
%dataΪ������������ݼ�
%AΪʹ�ö���ֲ��ų���˳��
%resultΪʹ��IFS��������ѡ��֮��ó��Ķ�Ӧ��׼ȷ��


data=res
num=336
n = randperm(size(data,1));
%% 
% 2. ѵ��������640������
data = data(n(1:num),:);
label = l3(n(1:num),:);

label=l3

features=size(data,2)
record=ones(features,2)
F=[];

%�����ݰ���ʹ�ö���ֲ��ó���˳���������
for j=1:features
    E=data(:,A(j));  
    F=[F E];
end
sortdata=F;
F1=[];
  
 for j=1:350
    j
    record(j,1)=j
    E=sortdata(:,j);
    F1=[F1 E];
    selectedfeature=F1    
    sum=0;
h=size(data,1);
res=ones(2,1);
for k=1:2
    indices = crossvalind('Kfold', h, 5);
    Allaccu=[];
    for i =1:5
        testdata=(indices == i);
        traindata=~testdata;
        test_data=selectedfeature(testdata,:);
        train_data=selectedfeature(traindata,:);

        testlabel=(indices == i);
        trainlabel=~testlabel;
        test_label=label(testlabel,:);
        train_label=label(trainlabel,:);

        [mtrain,ntrain]=size(train_data);
        [mtest,ntest]=size(test_data);
        dataset=[train_data;test_data];
        [dataset_scale,ps]=mapminmax(dataset',-1,1);
        dataset_scale=dataset_scale';
        train_data1=dataset_scale(1:mtrain,:);
        test_data1=dataset_scale((mtrain+1):(mtrain+mtest),:);
        model=svmtrain(train_label,train_data1,cmd);
        [predictlabel,accuracy,decision_values]=svmpredict(test_label,test_data1,model);
        Allaccu(i)=accuracy(1,1);
        sum=0;
        for i=1:length(Allaccu)
        sum=sum+Allaccu(i);
        end
        aveacc=sum/length(Allaccu);
    end
    res(k)=aveacc;
end
        record(j,2)=mean(res(:))
end

% figure
%  x=1:1:features;%x���ϵ����ݣ���һ��ֵ�������ݿ�ʼ���ڶ���ֵ��������������ֵ������ֹ
%  yvalue=record(:,2)
%  plot(x,yvalue,'-b','linewidth',2); %���ԣ���ɫ�����
% axis([0,features,0,100])  %ȷ��x����y���ͼ��С
% set(gca,'XTick',[0:128:features]) %x�᷶Χ1-6�����1
% set(gca,'YTick',[0:5:100]) %y�᷶Χ0-700�����100
% xlabel('Mumber of Pentamers')  %x����������
% ylabel('Overall accuracy(%)') %y����������
% %  text(x(226),yvalue(226),['(',num2str(x(226)),',',nu  m2str(yvalue(226)),')']);
%  [max_a,index]=max(yvalue)
end