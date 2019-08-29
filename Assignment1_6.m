%%%K-means Clustering
X = xlsread('C:\Users\HP\Desktop\KaunsiElectivesLenaHai\Neural Network and fuzzy logic BITS F312\data2.xlsx');
n = input('Enter number of iterations: '); %In this case, 100
[v, c] = kMeansClustering(X,2,n); %Calling the function for k-means clustering
classColors = zeros(size(X,1), 3); %Array to assign class colors to each data element
markerSizes = zeros(size(X,1), 1); %Array to assign marker size to each data element
for row = 1 : size(X,1)
if c(row)==1
% Class 1 = blue
classColors(row,:) = [0, 0, 1];
markerSizes(row) = 20;
else
% Class 2 = red
classColors(row, :) = [1, 0, 0];
markerSizes(row) = 20;
end
end
clear row
%Class vs feature values 1
figure;
scatter(X(:,1),c, markerSizes, classColors);
hold on; scatter(v(:,1),1:2,40,[0 1 0],'LineWidth',2)
%Class vs feature values 2
figure;
scatter(X(:,2),c, markerSizes, classColors);
hold on; scatter(v(:,2),1:2,40,[0 1 0],'LineWidth',2)
%Class vs feature values 3
figure;
scatter(X(:,3),c, markerSizes, classColors);
hold on; scatter(v(:,3),1:2,40,[0 1 0],'LineWidth',2)
%Class vs feature values 4
figure;
scatter(X(:,4),c, markerSizes, classColors);
hold on; scatter(v(:,4),1:2,40,[0 1 0],'LineWidth',2)
%Function to perform k - means Clustering
function [v, c] = kMeansClustering(X,K,n)
[row, col] = size(X);
v = zeros(K,col); %Cluster centres
c = zeros(row,1); %Array to assign clusters to each element in input
u = zeros(K,1); %Array to store number of elements in each cluster
p = randperm(size(X,1),K); %Shuffling data
for i = 1:K
v(i,:) = X(p(i),:); %Assigning K random values to initial centres
end
for i = 1:n
d = zeros(row,K); %Array to store distances
for j=1:row
for k = 1:K
%Find the distance between cluster center and the data point
d(j,k) = sqrt(sum((X(j,:)-v(k,:)).^2));
end
%Find the cluster to which the data point belongs
c(j) = find(d(j,:)==min(d(j,:)),1);
end
for k=1:K
%Calculating mean of cluster elements to recalculate new cluster centers
u(k) = sum(c(:)==k);
v(k,:) = sum(X(find(c(:)==k),:))/u(k);
end
end
end
