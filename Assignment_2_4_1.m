clc;
clear all;
close all;

data = xlsread('dataset.xlsx');
data(:,1:end-1) = (data(:,1:end-1)-mean(data(:,1:end-1)))./std(data(:,1:end-1));

X = data(:,1:end-1); %Inputs
Y = data(:,end); %Target outputs

for i=1:size(Y,1)
    if Y(i)==0
        btrainY(i,:)=[1,0];
    else
        btrainY(i,:)=[0,1];
    end
end


for i=1:size(Y,1)
    [~,l(:,i)]=max(btrainY(i,:));%index at which the maximum value of Y(i) occurs
end

c=cvpartition(l,'KFold',5);
for j=1:c.NumTestSets
    trIdx=c.training(j);
    teidx=c.test(j);
    xtr=X(trIdx,:);
    ytr=btrainY(trIdx,:);
    xte=X(teidx,:);
    yte=btrainY(teidx,:);
    xtrain=xtr;
    ytrain=ytr;
    xtest=xte;
    ytest=yte;
    
    tic
    NumberofHiddenNeurons=300;
    NumberofTrainingData=size(xtrain,1);
    NumberofTestingData=size(xtest,1);
    NumberofInputNeurons=size(xtrain,2);
    randommat=randn(NumberofInputNeurons+1,NumberofHiddenNeurons);
    
    tempH=[ones(size(xtrain,1),1) xtrain]*randommat;
    
    H=tanh(tempH);
    [m,n]=size(H);
    
    OutputWeight=pinv(H)*ytrain;
    
    testH=[ones(size(xtest,1),1) xtest]*randommat;
    
    Ht=tanh(testH);
    yn=Ht*OutputWeight;
    
    for i=1:size(xtest,1)
        [~, lp(:,i)]=max(yn(i,:));
        [~, lt(:,i)]=max(ytest(i,:));
    end
    
    [cmt,order]=confusionmat(lp,lt);
    display(cmt);
    IA=zeros(1,2);
    
    OA=0;
    for i=1:2
        IA(i)=cmt(i,i)/sum(cmt(i,:));
        OA=OA+cmt(i,i);
    end
    OA=OA/sum(cmt(:));
    toc
    display(IA);
    display(OA);
    
    
end
