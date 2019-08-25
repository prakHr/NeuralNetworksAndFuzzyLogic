clc;%clear command window
clear all;%clear alls the variable in matlab workspace
close all;%removes all variables from workspace
table=xlsread('data.xlsx');
X=ones(size(table,1),1);
X=[X table];
%normalization of inputs
X(:,2)=(X(:,2)-mean(X(:,2)))/std(X(:,2));
X(:,3)=(X(:,3)-mean(X(:,3)))/std(X(:,3));
X(:,4)=[];
%normalization of outputs
y=(table(:,3)-mean(table(:,3)))/std(table(:,3));
%initial weights
w=[0 0 0];
%clearing table
clear table;
%learning rate
alpha=0.01;
%No of iterations
k=input('Enter the number of iterations: ');
%------------------------------------------------------------------------------------------
%initiating
J=zeros(k+1,1);%one col of number of iterations 
W=zeros(k+1,3);%three col of number of iterations 
%evaluating first cost function
J(1)=evalcostfun(X,y,w);
%initial value of weights
W(1,1)=w(2);
W(1,2)=w(3);
%------------------------------------------------------------------------------------------
for i=1:k
    %evaluate hypothesis
    h=X*w'-y;
    for j=1:3
        w(j)=w(j)-alpha*(h'*X(:,j));
    end
    W(i+1,1)=w(2);
    W(i+1,2)=w(3);
    J(i+1)=evalcostfun(X,y,w);
end

plot(0:k,J);%Plotting cost function
xlabel('Iterations');
ylabel('cost function');
title('Assigment 1a');
grid on;

function J=evalcostfun(X,y,w)
J=norm(X*w'-y)^2;
end