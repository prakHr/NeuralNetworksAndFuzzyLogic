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
        v(k,:) = sum(X(c(:)==k,:))/u(k);
    end
end
end