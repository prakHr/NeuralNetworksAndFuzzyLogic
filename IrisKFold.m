clc;
clear all;
close all;
load iris_dataset.mat;


X=irisInputs';
Y=irisTargets';
for i=1:size(Y,1)
    [~,l(:,i)]=max(Y(i,:));%index at which the maximum value of Y(i) occurs
end

c=cvpartition(l,'KFold',5);
for j=1:c.NumTestSets
    trIdx=c.training(j);
    teidx=c.test(j);
    xtr=X(trIdx,:);
    ytr=Y(trIdx,:);
    xte=X(teidx,:);
    yte=Y(teidx,:);
    xtrain=xtr;
    ytrain=ytr;
    xtest=xte;
    ytest=yte;
    
    tic
    NumberofHiddenNeurons=30;
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
    cmt
    order
    summ=sum(cmt(:));
    %test_acc=(cmt(1,1)+cmt(2,2)/summ)*100;
    %test_sen=(cmt(1,1)/(cmt(1,1)+cmt(1,2)))*100;
    %test_spe=(cmt(2,2)/(cmt(2,2)+cmt(2,1)))*100;
    toc
end
