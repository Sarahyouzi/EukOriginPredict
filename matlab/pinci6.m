function [data1] = pinci6(data,allsequence)
%PINCI 此处显示有关此函数的摘要
%   此处显示详细说明
    data1=[]
    r1=size(data,1);      %核小体的样本的个数（即列数）
   c=300-6+1;
    r2=size(allsequence,1)
    for i=1:r1%r1是样本集的行数
        F=[];
        for k=1:r2%r2是AGTC所有排列组合的行数
            m=0;
            for j=1:c%c是样本集的列数
                 if [data(i,j) data(i,j+1) data(i,j+2) data(i,j+3) data(i,j+4) data(i,j+5)]==[allsequence(k,1) allsequence(k,2) allsequence(k,3) allsequence(k,4) allsequence(k,5) allsequence(k,6)]
                  m=m+1;        
                 end
            end
            F=[F m];
        end
         F=F/c
        data1=[data1;F];
    end

end

