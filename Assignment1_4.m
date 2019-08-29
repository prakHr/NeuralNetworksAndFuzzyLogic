table1 = xlsread('data.xlsx');
%Input feature matrix normalization
X(:,1:2) = (table1(:,1:2)-mean(table1(:,1:2)))./std(table1(:,1:2));
%Output vector normalization
y = (table1(:,3)-mean(table1(:,3)))/std(table1(:,3));
clear table1
w = (X'*X)\X'*y; %Weight evaluation using vectorised linear regression
w_gd = [1.13e-15; 0.07807; 0.36055]; %Weights from linear regression - batch gradient descent
w_sgd = [-0.0177; 0.09499; 0.3559]; %Weights from linear regression - stochastic gradient descent
e1 = sqrt(sum(([0; w]-w_gd).^2)); %Error with respect to batch gradient descent
e2 = sqrt(sum(([0; w]-w_sgd).^2)); %Error with respect to stochastic gradient descent
