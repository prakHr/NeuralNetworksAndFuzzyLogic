clc;
clear all;
close all;

data = xlsread('dataset.xlsx');
%Input normalization
data(:,1:end-1) = (data(:,1:end-1)-mean(data(:,1:end-1)))./std(data(:,1:end-1));
X = data(:,1:end-1); %Inputs
Y = data(:,end); %Target outputs

for i=1:length(Y)
    if Y(i)==1
        z(i,:) = [1 0];
    else
        z(i,:) = [0 1];
    end
end


%Randomly divide the dataset into training (70%) and testing (30%) set
%p = randperm(length(Y));
C = cvpartition(Y,'HoldOut',0.3);

    
tr = C.training;
te = C.test;
xTrain = X(tr,:);
xTest = X(te,:);
yTrain = z(tr,:);
yTest = z(te,:);


X=X';
Y=Y';

xTrain=xTrain';
yTrain=yTrain';

xTest=xTest';
yTest=yTest';

hiddenSize1 = 100;
autoenc1 = trainAutoencoder(xTrain,hiddenSize1, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false,...
    'DecoderTransferFunction','logsig',...
    'EncoderTransferFunction','logsig');

feat1 = encode(autoenc1,xTrain);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',100, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false,...
    'DecoderTransferFunction','logsig',...
    'EncoderTransferFunction','logsig');
feat2 = encode(autoenc2,feat1);

softnet = trainSoftmaxLayer(feat2,yTrain,'MaxEpochs',400);

stackednet = stack(autoenc1,autoenc2,softnet);

stackednet = train(stackednet,xTrain,yTrain);

yPredicted = stackednet(xTest);%2 X 644 double

yPredicted=yPredicted';%644 X 2 double
xTest=xTest';
for i=1:size(yPredicted,1)
    if yPredicted(i,1)>0.5
        y(i)=1;
    else
        y(i)=0;
    end

end


C1 = cvpartition(y,'HoldOut',0.3);
tr = C1.training;
te = C1.test;
xtrain = xTest(tr,:);
xtest = xTest(te,:);
ytrain = yPredicted(tr,:);
ytest = yPredicted(te,:);

tic
NumberofHiddenNeurons=100;
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
IA=zeros(1,size(cmt,1));

OA=0;
for i=1:size(cmt,1)
    IA(i)=cmt(i,i)/sum(cmt(i,:))*100;
    OA=OA+cmt(i,i);
end
OA=OA/sum(cmt(:))*100;
toc
display(IA);
display(OA);
