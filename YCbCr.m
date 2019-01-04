
clear;
clc;
feature=zeros(200,6);
for x =48:1:95
    temp=['F:\图片处理\Picture\干扰\',num2str(x),'\out_2.jpg'];
    img = imread(temp);%图像读入
    % figure,imshow(img);      %显示原图像
    ycbcr = rgb2ycbcr(img);  %rgb to ycbcr 函数
    % figure,imshow(ycbcr);
    fea=zeros(1,6);
    y=ycbcr(:,:,1);% 125 130
    y1=size(y(y>100));
    [m,n]=find(y>100);

    cb=ycbcr(:,:,2);
    cb1=double(cb(sub2ind(size(cb),m,n)));%提取Cb中Y》100的元素
    fea(1,1)=mean(cb1); %均值
    fea(1,2)=std(cb1,0); %标准差
    fea(1,3)=median(cb1); %中值

    cr=ycbcr(:,:,3);
    cr1=double(cr(sub2ind(size(cr),m,n)));%提取Cr中Y》100的元素
    fea(1,4)=mean(cr1); %均值
    fea(1,5)=std(cr1,0); %标准差
    fea(1,6)=median(cr1); %中值
    feature(x,1:6)=fea;
end