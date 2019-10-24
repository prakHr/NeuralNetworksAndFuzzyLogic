clear; clc;
close all;
data = xlsread('dataset.xlsx');
%Input normalization
data(:,1:end-1) = (data(:,1:end-1)-mean(data(:,1:end-1)))./std(data(:,1:end-1));
X = data(:,1:end-1); %Inputs
Y = data(:,end); %Target outputs
%Converting target output to two output neurons
for i=1:length(Y)
    if Y(i)==1
        z(i,:) = [1 0];
    else
        z(i,:) = [0 1];
    end
end
%Randomly divide the dataset into training (70%) and testing (30%) set
%p = randperm(length(Y));
C = cvpartition(Y,'KFold',5);

for j=1:C.NumTestSets
    tr = C.training(j);
    te = C.test(j);
    trainInput = X(tr,:);
    testInput = X(te,:);
    trainOutput = z(tr,:);
    testOutput = z(te,:);
    
    %trainInput = X(p(1:0.7*size(X,1)),:); trainOutput = z(p(1:0.7*size(X,1)),:);
    %testInput = X(p(0.7*size(X,1)+1:end),:); testOutput = z(p(0.7*size(X,1)+1:end),:);
    k = 600; %Hidden neurons
    [ind,c] = kmeans(trainInput,k); %kmeans clustering centres and indices
    n = zeros(k,1); %number of inputs belonging to each cluster
    for i=1:k
        n(i) = sum(ind(:)==i);
    end
    sigma = zeros(k,1); %standard deviation
    for i=1:k
        sigma(i) = norm(trainInput(ind(:)==i,:)-c(i))/n(i);
    end
    H=[];
    %MultiQuadratic Function
    SIGMA = sigma.^2;
    %Hidden layer matrix evaluation
    for i=1:length(trainOutput)
        for j=1:size(c,1)
            H(i,j) = (norm(trainInput(i,:)-c(j,:))^2+SIGMA(j)^2);
        end
    end
    W = pinv(H)*trainOutput; %Weight evaluation
    %Test data evaluation
    for i=1:length(testOutput)
        for j=1:size(c,1)
            Ht(i,j) = (norm(testInput(i,:)-c(j,:))^2+SIGMA(j)^2);
        end
    end
    yp = Ht*W; %Output evaluation
    %Class determination
    [~,pb]=max(testOutput,[],2);
    [~,pa]=max(yp,[],2);
    [cm, ~] = confusionmat(pa,pb); %calculating confusion matrix
    IA = zeros(1,2);
    OA = 0;
    for i = 1:2
        IA(i) = cm(i,i)/sum(cm(i,:)); %individual accuracy
    end
    for i=1:2
        OA= OA + cm(i,i);
    end
    
    OA = (OA/sum(cm(:)))*100; %overall accuracy
    
    display(cm);
    display(OA);
end