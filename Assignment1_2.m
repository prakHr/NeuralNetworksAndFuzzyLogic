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
k = input('Enter number of iterations: '); %In this case, 100
m = 50; %Number of training instances taken in each iteration
J = zeros(k*m+1,1); %Array to store cost for each iteration
W = zeros(k*m+1,3); %Array to store weights for each iteration
J(1) = evaluatecostfunction(X,y,w); %Calculating initial cost
W(1,1) = w(1);
W(1,2) = w(2);
W(1,3) = w(3);
for i=1:k
a = randperm(size(X,1),m); %Shuffling the data for each iteration
for j=1:m
h = (X(a(j),:)*w')-y(a(j)); %Hypothesis calculation
for b = 1:3
w(b) = w(b) - alpha*(h*X(a(j),b)); %Weight updates
W((i-1)*m+j+1,b) = w(b); %Storing weights in array
end
J((i-1)*m+j+1) = evaluatecostfunction(X,y,w); %Calculating cost
end
end
plot(0:k*m,J) %Plotting cost vs number of iterations
xlabel('No of iterations');ylabel('Cost Function');
clear i j a b h
w1 = 0.4:-0.001:-0.2;
w2 = 0.8:-0.005:-0.2;
J1 = zeros(length(w1),length(w2));
for i=1:length(w1)
for j=1:length(w2)
J1(i,j) = evaluatecostfunction(X,y,[0 w1(i) w2(j)]);
end
end
%Stochastic gradient descent - cost vs w1 and w2
figure; plot3(W(:,2),W(:,1),J,'color','r') %3D plot
figure; contour(w2,w1,J1); hold on; plot(W(:,3),W(:,2),'color','r') %Contour plot
clear i j
%Function to evaluate cost function for linear regression
function J = evaluatecostfunction(X,y,w)
J = 0;
for i=1:size(X,1)
J = J + ((w*X(i,:)')-y(i))^2;
end
J = 0.5*J;
end
