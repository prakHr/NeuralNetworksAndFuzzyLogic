clc; clear; close all;
x = [-1 -1 1 1; -1 1 -1 1]; %Inputs
y = [-1 1 1 -1]; %Target outputs
w1 = [0.1 -0.1; -0.1 0.1]; %weights b/w input and hidden layer
b1 = [-0.1; -0.1]; %bias b/w input and hidden layer
w2 = [0.1 0.1]; %weights b/w hidden and output layer
b2 = 0.1; %bias b/w hidden and output layer

theta = 0; %Threshold
for i=1:size(x,2)
    a1=w1*x(:,i)+b1;
    h1=2*(a1>=theta)-1;
    a2=w2*h1+b2;
    h2=2*(a2>=theta)-1;
    e(i)=y(i)-h2;
end
plot(e);

