table1 = xlsread('data3.xlsx');
%Input normalization
table1(:,1:4) = (table1(:,1:4)-mean(table1(:,1:4))./std(table1(:,1:4)));
p = randperm(size(table1,1)); %shuffling data
train_data = table1(p(1:0.6*size(table1,1)),:);
%Dividing the data into training data (60%) and testing data (40%)
test_data = table1(p(0.6*size(table1,1)+1:end),:);
clear table1 p
t1 = []; t2 = []; %array to store data points belonging to classes 1 and 2
py1 = 0; py2 = 0; %to store probability of classes 1 and 2
for i = 1:size(train_data,1)
switch train_data(i,5)
case 1
t1 = [t1; train_data(i,:)]; py1 = py1+1;
case 2
t2 = [t2; train_data(i,:)]; py2 = py2+1;
end
end
py1 = py1/size(train_data,1); py2 = py2/size(train_data,1); %probability of the 2 classes
u1 = mean(t1(:,1:4))'; u2 = mean(t2(:,1:4))'; %mean of classes 1 and 2
E1 = t1(:,1:4)'*t1(:,1:4); %covariance matrix of class 1
E1_det = det(E1); %determinant of covariance matrix 1
E2 = t2(:,1:4)'*t2(:,1:4); %covariance matrix of class 2
E2_det = det(E2); %determinant of covariance matrix 2
yp = zeros(size(test_data,1),1);
for i = 1:size(test_data,1)
x = test_data(i,1:4)';
p_x_y1 = ((2*pi)^(-2))*(E1_det^(-0.5))*exp(-0.5*(x-u1)'*pinv(E1)*(x-u1));
p_x_y2 = ((2*pi)^(-2))*(E2_det^(-0.5))*exp(-0.5*(x-u2)'*pinv(E2)*(x-u2));
p_x_y = [p_x_y1/p_x_y2 py2/py1]; %calculating LRT
[~, yp(i)] = max(p_x_y); %determining class
end
TN = 0; %True negative
TP = 0; %True positive
FN = 0; %False negative
FP = 0; %False positive
for i = 1:size(yp,1)
switch yp(i)
case 1
if yp(i)==test_data(i,5)
TN = TN+1;
else
FN = FN+1;
end
case 2
if yp(i)==test_data(i,5)
TP = TP+1;
else
FP = FP+1;
end
end
end
ConfusionMatrix = [TN FP; FN TP]; Sensitivity = TP/(TP+FN); Specificity = TN/(TN+FP);
Accuracy = (TP+TN)/(TP+TN+FP+FN);
clear i FP FN TP TN p_x_y1 p_x_y2
