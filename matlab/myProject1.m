
%4��������������ӱ�ǩ
load('pseknc2.2.mat')
load('lable.mat')
data1=[l3 data]

for i=1:7
    for j=1:6
        a(i,j)=a(i,j)*3
    end
end




data=res
����
data=featureSelect(data,A,500,290)
���Ͻ�
data=featureSelect(data,A,5640,147)
��Ӭ
data=featureSelect(data,A,500,184)
С��
data=featureSelect(data,A,500,264)

%5��ʹ��python����������ѡ�񣬵õ��Ƽ����������
%6�������Ƽ���������ϣ�������������
%selectedfeature=featureSelect(data,A,1024,1021)  
%selectedfeature=data 

%7��ʹ��svm
% 1. �������ѵ�����Ͳ��Լ�
data=res
n = randperm(size(data,1));
%%
% 2. ѵ��������640������
train_matrix = data(n(1:336),:);
train_label = l3(n(1:336),:);

train_matrix = data;
train_label = l3;
train_matrix = train_matrix(n(1:2000),:);
train_label = train_label(n(1:2000),:);
%% III. ���ݹ�һ��
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';

%% IV. SVM����/ѵ��(RBF�˺���)
%%
% 1. Ѱ�����c/g��������������֤����
%[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
%�ڶ����ı䷶Χ
m=31
n=61
cg = zeros(m,n);
eps = 10^(-4);
v = 5;
bestc = 1;
bestg = 0.1;
bestacc = 0;
for i = 1:m
    i
    for j = 1:n
        j
        cmd = ['-v ',num2str(v),' -t 2',' -c ',num2str(2^c(i,j)),' -g ',num2str(2^g(i,j))];
         %-v������˵�����صĲ���model����һ��������
        cg(i,j) = svmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);%����õõ��ҳ���
            bestc = 2^c(i,j);%ͬʱ��¼c��g��ֵ
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            %���Ѿ��ҵ���ѣ������㾫��Ҫ��Ϊ�˼ӿ��ٶȣ���ô��cΪ��׼ȡһ��
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
save('cmd-comb2,8,11-300.mat','cmd')
%%
% 2. ����/ѵ��SVMģ��
model = svmtrain(train_label,Train_matrix,cmd);

test_matrix = data;
test_label = l3;
%% III. ���ݹ�һ��
[Test_matrix,PS] = mapminmax(test_matrix');
Test_matrix =Test_matrix';


%% V. SVM�������
[predict_label_1,accuracy_1,dec_value] = svmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,dec_value] = svmpredict(test_label,Test_matrix,model);
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];

%% VI. ��ͼ
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('��ʵ���','Ԥ�����')
xlabel('���Լ��������')
ylabel('���Լ��������')
string = {'���Լ�SVMԤ�����Ա�(RBF�˺���)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)  