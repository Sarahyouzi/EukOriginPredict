function [text] = textD(seq,pos)
%TEXTD 此处显示有关此函数的摘要
%tf 代表某个核苷酸在正样本中出现的频率
%dif 代表某个核苷酸在所有样本中的出现频率
%ti_dif代表最后的得分
%data代表每个序列中出现的核苷酸的频次

    load('pinci1.mat')
    pos=2380
    data1=data(1:pos,:)
    r1=size(data,1)
    r2=size(data,2)
   
    tf1=zeros(1,r2);
    for i=1:r2
        tf1(1,i)=sum(data1(:,i))/pos;
    end
    
    idf1=zeros(1,r2)
    for i=1:r1
        for j=1:r2
            if data(i,j)~=0
                idf1(1,j)=idf1(1,j)+1;
            end
        end
    end
    tf_idf1=zeros(1,r2)
    for i=1:r2
       tf_idf1(1,i)=tf1(1,i)*log2(r1/(idf1(1,i)+1))
    end
    
    
      load('pinci2.mat')
      data=data*300/299
      data2=data(1:pos,:)
    r2=size(data,2)
   
    tf2=zeros(1,r2);
    for i=1:r2
        tf2(1,i)=sum(data2(:,i))/pos;
    end
    
    idf2=zeros(1,r2)
    for i=1:r1
        for j=1:r2
            if data(i,j)~=0
                idf2(1,j)=idf2(1,j)+1;
            end
        end
    end
    tf_idf2=zeros(1,r2)
    for i=1:r2
       tf_idf2(1,i)=tf2(1,i)*log2(r1/(idf2(1,i)+1))
    end
    
     load('pinci3.mat')
      data=data*300/298
     data3=data(1:pos,:)
    r2=size(data,2)
   
    tf3=zeros(1,r2);
    for i=1:r2
        tf3(1,i)=sum(data3(:,i))/pos;
    end
    
    idf3=zeros(1,r2)
    for i=1:r1
        for j=1:r2
            if data(i,j)~=0
                idf3(1,j)=idf3(1,j)+1;
            end
        end
    end
    tf_idf3=zeros(1,r2)
    for i=1:r2
       tf_idf3(1,i)=tf3(1,i)*log2(r1/(idf3(1,i)+1))
    end
    
    
      load('pinci4.mat')
      data=data*300/297
     data4=data(1:pos,:)
    r2=size(data,2)
   
    tf4=zeros(1,r2);
    for i=1:r2
        tf4(1,i)=sum(data4(:,i))/pos;
    end
    
    idf4=zeros(1,r2)
    for i=1:r1
        for j=1:r2
            if data(i,j)~=0
                idf4(1,j)=idf4(1,j)+1;
            end
        end
    end
    tf_idf4=zeros(1,r2)
    for i=1:r2
       tf_idf4(1,i)=tf4(1,i)*log2(r1/(idf4(1,i)+1))
    end
     
     load('pinci5.mat')
      data=data*300/296
       data5=data(1:pos,:)
     
    r2=size(data,2)
   
    tf5=zeros(1,r2);
    for i=1:r2
        tf5(1,i)=sum(data5(:,i))/pos;
    end
    
    idf5=zeros(1,r2)
    for i=1:r1
        for j=1:r2
            if data(i,j)~=0
                idf5(1,j)=idf5(1,j)+1;
            end
        end
    end
    tf_idf5=zeros(1,r2)
    for i=1:r2
       tf_idf5(1,i)=tf5(1,i)*log2(r1/(idf5(1,i)+1))
    end
    
     load('pinci6.mat')
      data=data*300/295
       data6=data(1:pos,:)
    r2=size(data,2)
   
    tf6=zeros(1,r2);
    for i=1:r2
        tf6(1,i)=sum(data6(:,i))/pos;
    end
    
    idf6=zeros(1,r2)
    for i=1:r1
        for j=1:r2
            if data(i,j)~=0
                idf6(1,j)=idf6(1,j)+1;
            end
        end
    end
    tf_idf6=zeros(1,r2)
    for i=1:r2
       tf_idf6(1,i)=tf6(1,i)*log2(r1/(idf6(1,i)+1))
    end
    
    text=[tf_idf1 tf_idf2 tf_idf3 tf_idf4 tf_idf5 tf_idf6]
    text=text'
    n=(1:5460)
    n=n'
    data=[n text];
    save('tf_idf','data')
end


    