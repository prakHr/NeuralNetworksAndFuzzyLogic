clc;
clear all;
close all;
%load data4.mat;
data3 = xlsread('data4.xlsx');
x=data3(:,(1:7));
y=data3(:,8);

for j=1:size(x,2)
    x(:,j)=x(:,j)/max(abs(x(:,j)));
end

yu=unique(y);
[u,~]=size(yu);

%taking partition
c=cvpartition(y,'HoldOut',0.4);
trIdx=c.training;
teIdx=c.test;

xtr=x(trIdx,:);
xte=x(teIdx,:);

ytr=y(trIdx);
yte=y(teIdx);

T=300;
alpha=0.01;

[m1,n]=size(xtr);
[m2,n]=size(xte);

xtr=[ones(m1,1) xtr];
xte=[ones(m2,1) xte];

for i=1:u
    for j=1:m1
        if(ytr(j)==yu(i,:))
            ymtr(j,i)=0;
        else
            ymtr(j,i)=1;
        end
    end
    yp(:,i)=LOG_R(n,T,alpha,xtr,xte,ymtr(:,i));
end

[~,ypp]=min(yp,[],2);
[cm,order]=confusionmat(yte,double(ypp));
display(cm);

for i=1:length(order)
    IA(i)=cm(i,i)/sum(cm(i,:))*100;
end
display(IA);

OA=trace(cm)/sum(cm(:))*100;
display(OA);