%%%%Batch Gradient Descent
clc;
clear all;
close all;
table1 = xlsread('data.xlsx');
% X is the matrix containing feature vectors for all instances
X = ones(size(table1,1),1);
X = [X table1(:,1:2)];
% Inputs normalization
X(:,2) = (X(:,2)-mean(X(:,2)))/std(X(:,2));
X(:,3) = (X(:,3)-mean(X(:,3)))/std(X(:,3));
%Target outputs
y = (table1(:,3)-mean(table1(:,3)))/std(table1(:,3));
clear table1
w = [0 0 0]; %Initial weights
alpha = 0.001; %Learning rate
lambda = 0.4; %Regularisation parameter
k = input('Enter number of iterations: '); %In this case, 25
J = zeros(k+1,1); %Array to store cost for each iteration
W = zeros(k+1,2); %Array to store weigthts for each iteration
J(1) = evaluatecostfunction(X,y,w,lambda); %Calculating initial cost
W(1,1) = w(1);
W(1,2) = w(2);
for i=1:k
h = (X*w')-y; %Hypothesis calculation
for b=1:3
w(b) = w(b) - alpha*(h'*X(:,b))-0.5*alpha*lambda*sign(w(b)); %Weight updates
end
J(i+1) = evaluatecostfunction(X,y,w,lambda); %Calculating cost
W(i+1,1) = w(2);
W(i+1,2) = w(3);
end
plot(0:k,J) %Plotting cost vs number of iterations
clear b i h
w1 = 0.4:-0.001:-0.2;
w2 = 0.8:-0.005:-0.2;
J1 = zeros(length(w1),length(w2));
for i=1:length(w1)
for j=1:length(w2)
J1(i,j) = evaluatecostfunction(X,y,[0 w1(i) w2(j)],lambda);
end
end
%Plotting weights vs cost function
figure; plot3(W(:,2),W(:,1),J,'color','r') %Batch gradient descent – Cost vs w1 and w2
figure; contour(w2,w1,J1); hold on; plot(W(:,2),W(:,1),'color','r') %Contour plot
clear i j
%Function to evaluate cost function for least angle regression
function J = evaluatecostfunction(X,y,w,l)
J = 0;
for i=1:size(X,1)
J = J + ((w*X(i,:)')-y(i))^2;
end
J = J+(l*sum(abs(w)));
J = 0.5*J;
end