clc;clear;close all;
x=[-1 -1 1 1;-1 1 -1 1];%inputs
y=[-1 -1 -1 1];%target outputs
w=[0 0];%Initial weights
b=0;%initial bias
alpha=0.1;%learning rate
theta=0;%threshold
iter=10;%num of iterations
for k=1:iter
    for i=1:size(x,2)%size=4
        a(i)=w*x(:,i)+b%size(w)=1X2,size(x(:,i))=2X1
        h(i)=2*(a(i)>=theta)-1
        if h(i)~=y(i)
            w=w+alpha*y(i)*x(:,i)';
            b=b+alpha*y(i);
        end
    end
    e(k)=sum((y-h).^2);
end
plot(e)