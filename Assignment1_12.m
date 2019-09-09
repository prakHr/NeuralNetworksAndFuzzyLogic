clc,
clear all;
close all;
table1 = xlsread('data4.xlsx');
%normalization of data
table1(:,1:4) = (table1(:,1:4)-mean(table1(:,1:4))./std(table1(:,1:4)));
data1 = table1(1:50,:);%size(data1)=50X8,size(data2)=50X8,size(data3)=50X8
data2 = table1(51:100,:);
data3 = table1(101:end,:);
p = randperm(50); %shuffling data
%dividing data to 70% training data and 30% test data
train_data = [data1(p(1:0.7*size(data1,1)),:); data2(p(1:0.7*size(data2,1)),:); data3(p(1:0.7*size(data2,1)),:)];
test_data = [data1(p(0.7*size(data1,1)+1:end),:); data2(p(0.7*size(data2,1)+1:end),:); data3(p(0.7*size(data2,1)+1:end),:)];
%size(train_data)=105X8,test_data=45X8
clear data1 data2 data3 table1
t1 = [];
t2 = [];
t3 = [];
for i = 1:size(train_data,1)
    switch train_data(i,5)
        case 1
            t1 = [t1; train_data(i,:)];
        case 2
            t2 = [t2; train_data(i,:)];
        case 3
            t3 = [t3; train_data(i,:)];
    end
end
%calculating mean
u1 = mean(t1(:,1:4))';
u2 = zeros(4,1);
u3 = zeros(4,1);
%calculating covariance matrix and determinant of covariance matrix
E1 = t1(:,1:4)'*t1(:,1:4);
E1_det = det(E1);
E2 = zeros(4);
E2_det = det(E2);
E3 = zeros(4);
E3_det = det(E3);
yp = zeros(size(test_data,1),1);
for i = 1:size(test_data,1)
    x = test_data(i,1:4)';
%calculating p(x/yk)
    p_x_y1 = ((2*pi)^(-2))*(E1_det^(-0.5))*exp(-0.5*(x-u1)'*pinv(E1)*(x-u1));
    p_x_y2 = ((2*pi)^(-2))*(E2_det^(-0.5))*exp(-0.5*(x-u2)'*pinv(E2)*(x-u2));
    p_x_y3 = ((2*pi)^(-2))*(E3_det^(-0.5))*exp(-0.5*(x-u3)'*pinv(E3)*(x-u3));
    [~, yp(i)] = max([p_x_y1 p_x_y2 p_x_y3]); %calculating ML output
end
c = zeros(3); %calculating confusion matrix
for i = 1:size(yp,1)
    if test_data(i,5) == 1
        c(1,yp(i)) = c(1,yp(i))+1;
    elseif test_data(i,5) == 2
        c(2,yp(i)) = c(2,yp(i))+1;
    else
        c(3,yp(i)) = c(3,yp(i))+1;
    end
end
IA = zeros(1,3);
OA = 0;
for i = 1:3
    IA(i) = c(i,i)/sum(c(i,:)); %individual accuracy
    OA = OA + c(i,i);
end
OA = OA/sum(c(:)); %overall accuracy
ConfusionMatrix = c; %confusion matrix
clear p_x_y p_x_y1 p_x_y2 p_x_y3 py1 py2 py3 x c i yp_1 yp_2 yp_3 a
