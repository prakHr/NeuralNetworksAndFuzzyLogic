clc; clear; close all;
x = [-1 -1 1 1; -1 1 -1 1]; %Inputs
y = [1 1 1 -1]; %Target outputs
w = [0 0]; %Initial weight
b = 0; %Initial bias
alpha = 0.1;%Learning rate
theta = 0; %Threshold
iter = 10; %Number of iterations
for k=1:iter
for i=1:size(x,2)
a(i) = w*x(:,i)+b; h(i) = 2*(a(i)>=theta)-1;
if(h(i)~=y(i)) %Hebbian rule weight update
w = w + alpha*y(i)*x(:,i)'; b = b + alpha*y(i);
end
end
e(k) = sum((y-h).^2); %Cost Function
end
plot(e) %Cost Function vs Number of Iterations