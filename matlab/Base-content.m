function [data] = GC(seq)
%PINCI �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
    data=[]
    r1=size(seq,1);      %��С��������ĸ�������������
    for i=1:r1%r1��������������
        numG=0;
        numC=0;
        numA=0;
        numT=0;
        for j=1:300%r2��AGTC����������ϵ�����
            if seq(i,j)=='C'
               numC=numC+1;        
            end
            if seq(i,j)=='G'
               numG=numG+1;        
            end
            if seq(i,j)=='A'
               numA=numA+1;        
            end
            if seq(i,j)=='T'
               numT=numT+1;        
            end
        end
           GC_skew=(numG-numC)/(numG+numC)
           GC_profile=(numG+numC)/300
           AT_skew=(numA-numT)/(numA+numT)
           AT_profile=(numA+numT)/300
           data=[data;GC_skew GC_profile AT_skew AT_profile]
    end
end

