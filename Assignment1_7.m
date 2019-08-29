X = xlsread('data3.xlsx');
X(:,1:4) = (X(:,1:4)-mean(X(:,1:4)))./std(X(:,1:4)); %Input normalization
X(:,5) = X(:,5)-1; %Converting classes 1 and 2 to 0 and 1
p = randperm(size(X,1)); %Shuffling data
%Dividing the data into training data (60%) and testing data (40%)
train_data = [ones(0.6*size(X,1),1) X(p(1:0.6*size(X,1)),:)];
test_data = [ones(0.4*size(X,1),1) X(p(0.6*size(X,1)+1:end),:)];
clear p X
w = zeros(1,size(train_data,2)-1); %Weights initialization
alpha = 0.01; %Learning rate
k = 10; %Number of iterations
y = train_data(:,6); %Target classes for training data
for i=1:k
g = logsig(w*train_data(:,1:5)')'; %Hypothesis calculation
%Weights update
for j = 1:size(train_data,2)-1
w(j) = w(j)-alpha*sum((y.*(1-g)+(y-1).*g).*train_data(:,j));
end
end
clear i j g y
yp = logsig(w*test_data(:,1:5)')'; %Hypothesis calculation using test data
yp = 1*(yp<0.5); %Class prediction
y_1 = test_data(:,6); %Target classes for testing data
TN = 0; %True negative
TP = 0; %True positive
FN = 0; %False negative
FP = 0; %False positive
for i = 1:size(yp,1)
switch yp(i)
case 0
if yp(i)==y_1(i)
TN = TN+1;
else
FN = FN+1;
end
case 1
if yp(i)==y_1(i)
TP = TP+1;
else
FP = FP+1;
end
end
end
ConfusionMatrix = [TN FP; FN TP]; %Calculation of confusion matrix
%Sensitivity - proportion of abnormal episodes that are accurately classified as abnormal
Sensitivity = TP/(TP+FN);
%Specificity - proportion of normal episodes that are accurately classified as normal
Specificity = TN/(TN+FP);
%Accuracy - proportion of episodes that are accurately classified as normal and abnormal
Accuracy = (TP+TN)/(TP+TN+FP+FN);
\
