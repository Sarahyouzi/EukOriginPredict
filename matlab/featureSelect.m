function [selectedArray] = featureSelect(data,A1,arraySize,a)
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
F=[];
for j=1:arraySize
    E=data(:,A1(j));  
    F=[F E];
end
sortdata=F;
F1=[];
for j=1:a
    E=sortdata(:,j);
    F1=[F1 E];
end
    selectedArray=F1;%data4�Ǿ���������ѡ��֮�����µĸ�������������

end
    
