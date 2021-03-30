
%4、给样本数据添加标签
load('pseknc2.2.mat')
load('lable.mat')
data1=[l3 data]

for i=1:7
    for j=1:6
        a(i,j)=a(i,j)*3
    end
end




data=res
人类
data=featureSelect(data,A,500,290)
拟南芥
data=featureSelect(data,A,5640,147)
果蝇
data=featureSelect(data,A,500,184)
小鼠
data=featureSelect(data,A,500,264)

%5、使用python对特征进行选择，得到推荐的特征组合
%6、根据推荐的特征组合，来整理特征集
%selectedfeature=featureSelect(data,A,1024,1021)  
%selectedfeature=data 

%7、使用svm
% 1. 随机产生训练集和测试集
data=res
n = randperm(size(data,1));
%%
% 2. 训练集——640个样本
train_matrix = data(n(1:336),:);
train_label = l3(n(1:336),:);

train_matrix = data;
train_label = l3;
train_matrix = train_matrix(n(1:2000),:);
train_label = train_label(n(1:2000),:);
%% III. 数据归一化
[Train_matrix,PS] = mapminmax(train_matrix');
Train_matrix = Train_matrix';

%% IV. SVM创建/训练(RBF核函数)
%%
% 1. 寻找最佳c/g参数——交叉验证方法
%[c,g] = meshgrid(-10:0.2:10,-10:0.2:10);
%第二步改变范围
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
         %-v参数，说明返回的不是model而是一个标量数
        cg(i,j) = svmtrain(train_label,Train_matrix,cmd);     
        if cg(i,j) > bestacc
            bestacc = cg(i,j);%把最好得到找出来
            bestc = 2^c(i,j);%同时记录c和g的值
            bestg = 2^g(i,j);
        end        
        if abs( cg(i,j)-bestacc )<=eps && bestc > 2^c(i,j) 
            %当已经找到最佳，且满足精度要求，为了加快速度，那么以c为标准取一个
            bestacc = cg(i,j);
            bestc = 2^c(i,j);
            bestg = 2^g(i,j);
        end               
    end
end
cmd = [' -t 2',' -c ',num2str(bestc),' -g ',num2str(bestg)];
save('cmd-comb2,8,11-300.mat','cmd')
%%
% 2. 创建/训练SVM模型
model = svmtrain(train_label,Train_matrix,cmd);

test_matrix = data;
test_label = l3;
%% III. 数据归一化
[Test_matrix,PS] = mapminmax(test_matrix');
Test_matrix =Test_matrix';


%% V. SVM仿真测试
[predict_label_1,accuracy_1,dec_value] = svmpredict(train_label,Train_matrix,model);
[predict_label_2,accuracy_2,dec_value] = svmpredict(test_label,Test_matrix,model);
result_1 = [train_label predict_label_1];
result_2 = [test_label predict_label_2];

%% VI. 绘图
figure
plot(1:length(test_label),test_label,'r-*')
hold on
plot(1:length(test_label),predict_label_2,'b:o')
grid on
legend('真实类别','预测类别')
xlabel('测试集样本编号')
ylabel('测试集样本类别')
string = {'测试集SVM预测结果对比(RBF核函数)';
          ['accuracy = ' num2str(accuracy_2(1)) '%']};
title(string)  