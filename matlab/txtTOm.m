% load('C:\Users\Lv\Desktop\data\Human Activity\train_label.txt')
% save('F:\MATLAB\R2015b\bin\Adaboost\train_label1.mat','train_label')

% .fscore �÷־����һ����
A=textread('data1_tfidf350.txt.fscore.txt','%s','delimiter',':');
a=size(A,1)/2-1;     %����ǩ������ѵ��������������2�ټ�1��Ҳ�������ݼ���һ�����Ŀa
C=[];
for i=0:a
    D11=A(2*i+1,:);  %��������
    C=[C D11];       %ˮƽ����
end

A=str2double(C);    %ȡ���ݼ���һ�룬��ȡһ�����ݼ��������У�str����ת��double����

% for i=1:3
%     add(1,i)=1029+i 
% end
% A34=[add A3]

%ѡȡ�ض���ά�ȣ�����C��
% a=size(A,1)/2-1;
% C=[];
% for i=0:a
%     D11=A(2*i+1,:);
%     C=[C D11]; 
% end
% C1=char(C)
% D=double(1:256)
% % A1=str2double(C);
% %�������ݣ��������Ϊ��ֵ
% for i=1:size(C1,1)
%     str=''
%     for j=1:4
%         if C1(i,j)==':'
%             break
%         end
%         str=strcat(str,C1(i,j))
%     end
%     D(i)=str2double(str)
% end
% sortdata=D