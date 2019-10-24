clc;clear all;close all;
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
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.004, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.15, ...
    'ScaleData', false,...
    'DecoderTransferFunction','logsig',...
    'EncoderTransferFunction','logsig');

feat1 = encode(autoenc1,xTrain);

hiddenSize2 = 50;
autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
    'MaxEpochs',400, ...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false,...
    'DecoderTransferFunction','logsig',...
    'EncoderTransferFunction','logsig');
feat2 = encode(autoenc2,feat1);

hiddenSize3 = 25;
autoenc3= trainAutoencoder(feat2,hiddenSize3,...
    'MaxEpochs',400,...
    'L2WeightRegularization',0.002, ...
    'SparsityRegularization',4, ...
    'SparsityProportion',0.1, ...
    'ScaleData', false,...
    'DecoderTransferFunction','logsig',...
    'EncoderTransferFunction','logsig');%purelin
feat3 = encode(autoenc3,feat2);
softnet = trainSoftmaxLayer(feat3,yTrain,'MaxEpochs',400);

%view(autoenc1);
%view(autoenc2);
%view(softnet);
stackednet = stack(autoenc1,autoenc2,autoenc3,softnet);

%stackednet.trainFcn='trainlm';'trainscg'
%stackednet.trainParam.epochs=10;

stackednet = train(stackednet,xTrain,yTrain);
%view(stackednet);

w = stackednet(xTest);
%plotconfusion(yTest,w);

w=w';
yTest=yTest';

for i=1:size(yTest,1)
    [~, lp(:,i)]=max(w(i,:));
    [~, lt(:,i)]=max(yTest(i,:));
end

[cmt,order]=confusionmat(lp,lt);


test_acc=(cmt(1,1)+cmt(2,2)/sum(cmt(:)))*100;
test_sen=(cmt(1,1)/(cmt(1,1)+cmt(1,2)))*100;
test_spe=(cmt(2,2)/(cmt(2,2)+cmt(2,1)))*100;
display(test_acc);
display(test_sen);
display(test_spe);
IA=zeros(1,size(cmt,1));
OA=0;
for i=1:size(cmt,1)
    IA(i)=cmt(i,i)/sum(cmt(i,:));
    OA=OA+cmt(i,i);
end
OA=OA/sum(cmt(:));
display(cmt);
display(IA);
display(OA);

    