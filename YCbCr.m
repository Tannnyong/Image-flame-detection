
clear;
clc;
feature=zeros(200,6);
for x =48:1:95
    temp=['F:\ͼƬ����\Picture\����\',num2str(x),'\out_2.jpg'];
    img = imread(temp);%ͼ�����
    % figure,imshow(img);      %��ʾԭͼ��
    ycbcr = rgb2ycbcr(img);  %rgb to ycbcr ����
    % figure,imshow(ycbcr);
    fea=zeros(1,6);
    y=ycbcr(:,:,1);% 125 130
    y1=size(y(y>100));
    [m,n]=find(y>100);

    cb=ycbcr(:,:,2);
    cb1=double(cb(sub2ind(size(cb),m,n)));%��ȡCb��Y��100��Ԫ��
    fea(1,1)=mean(cb1); %��ֵ
    fea(1,2)=std(cb1,0); %��׼��
    fea(1,3)=median(cb1); %��ֵ

    cr=ycbcr(:,:,3);
    cr1=double(cr(sub2ind(size(cr),m,n)));%��ȡCr��Y��100��Ԫ��
    fea(1,4)=mean(cr1); %��ֵ
    fea(1,5)=std(cr1,0); %��׼��
    fea(1,6)=median(cr1); %��ֵ
    feature(x,1:6)=fea;
end