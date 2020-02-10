clc;
clear all;
close all;

load data_for_cnn.mat;
load class_label.mat;

for j=1:1000
    ecg_in_window(:,j)=ecg_in_window(:,j)/max(abs(ecg_in_window(:,j)));
end
height = 1;
width = 1000;
channels = 1;

C = cvpartition(label,'HoldOut',0.3);
tr = C.training;
te = C.test;
Xtr = ecg_in_window(tr,:);
Xte = ecg_in_window(te,:);
Ytr = label(tr,:);
Yte = label(te,:);

Xtr = reshape(Xtr,[height width channels 700]);
Xte = reshape(Xte,[height width channels 300]);

Ytr = categorical(Ytr);
Yte = categorical(Yte);

layers = [
    imageInputLayer([height width channels])
    
    convolution2dLayer([1, 25],10)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([1 20])
    
     fullyConnectedLayer(20)
  
    fullyConnectedLayer(2)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...
    'MaxEpochs',200, 'Plots','training-progress');


net = trainNetwork(Xtr,Ytr,layers,options);

YPred = classify(net,Xte);
YValidation = Yte;

accuracy = sum(YPred == YValidation)/numel(YValidation);