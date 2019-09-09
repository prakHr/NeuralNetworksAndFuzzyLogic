table1 = xlsread('data4.xlsx');
%normalization of data
table1(:,1:4) = (table1(:,1:4)-mean(table1(:,1:4)))./std(table1(:,1:4));
table1 = [ones(length(table1),1) table1];
p = randperm(length(table1));
%dividing data to 70% training data and 30% test data
train_data = table1(p(1:0.7*size(table1,1)),:);
test_data = table1(p(0.7*size(table1,1)+1:end),:);
clear table1 p
t1 = [];
t2 = [];
t3 = [];
py1 = 0;
py2 = 0;
py3 = 0;
%dividing data based on the output class
for i = 1:length(train_data)
    switch train_data(i,end)
        case 1
            t1 = [t1; train_data(i,:)]; py1 = py1+1;
        case 2
            t2 = [t2; train_data(i,:)]; py2 = py2+1;
        case 3
            t3 = [t3; train_data(i,:)]; py3 = py3+1;
    end
end
py1 = py1/size(train_data,1); %probability of class 1
py2 = py2/size(train_data,1); %probability of class 2
py3 = py3/size(train_data,1); %probability of class 3
%calculating mean
u1 = mean(t1(:,1:end-1))'; 
u2 = mean(t2(:,1:end-1))';
u3 = mean(t3(:,1:end-1))';
%calculating covariance matrix and determinant of covariance matrix
E1 = t1(:,1:end-1)'*t1(:,1:end-1);
E1_det = det(E1);

E2 = t2(:,1:end-1)'*t2(:,1:end-1);
E2_det = det(E2);

E3 = t3(:,1:end-1)'*t3(:,1:end-1);
E3_det = det(E3);

yp = zeros(size(test_data,1),1);
for i = 1:size(test_data,1)
    x = test_data(i,1:end-1)';
    %calculating p(x/yk)
    p_x_y1 = ((2*pi)^(-2))*E1_det^(-0.5)*exp(-0.5*((x-u1)'/E1)*(x-u1));
    p_x_y2 = ((2*pi)^(-2))*E2_det^(-0.5)*exp(-0.5*((x-u2)'/E2)*(x-u2));
    p_x_y3 = ((2*pi)^(-2))*E3_det^(-0.5)*exp(-0.5*((x-u3)'/E3)*(x-u3));
    p_x_y = [p_x_y1*py1 p_x_y2*py2 p_x_y3*py3]; %calculating p(yk/x)
    [~, yp(i)] = max(p_x_y); %calculating MAP output
end
c = zeros(3); %calculating confusion matrix
for i = 1:size(yp,1)
    c(test_data(i,end),yp(i)) = c(test_data(i,end),yp(i))+1;
end
IA = zeros(1,3); OA = 0;
for i = 1:3
    IA(i) = c(i,i)/sum(c(i,:)); %individual accuracy
    OA = OA + c(i,i);
end
OA = OA/sum(c(:)); %overall accuracy
ConfusionMatrix = c; %confusion matrix
clear p_x_y p_x_y1 p_x_y2 p_x_y3 x c i yp_1 yp_2 yp_3 a
