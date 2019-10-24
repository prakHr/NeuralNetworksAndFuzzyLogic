clc;
clear all;
close all;

Q1 = 7;
Q2 = 3;

m = 3;
mu = 0.75;
a = 0.0001;
T = 1000;
MSETarget = 1e-20;

data = xlsread('dataset.xlsx');
data(:,1:end-1) = (data(:,1:end-1)-mean(data(:,1:end-1)))./std(data(:,1:end-1));

X = data(:,1:end-1); %Inputs
Y = data(:,end); %Target outputs

for i = 1:length(Y)
    if(Y(i)==1)
        Zy(i,:) = [0 1];   
    else  
        Zy(i,:) = [1 0];
    end
end

C = cvpartition(Y,'HoldOut',0.3);
tr = C.training;
te = C.test;
Xtr = X(tr,:);
Xte = X(te,:);
Ztr = Zy(tr,:);
Zte = Zy(te,:);

[p,N] = size(Xtr);
[p2,N2] = size(Xte);
bias = -1;
Xtr = [bias*ones(p,1) Xtr];
Xte = [bias*ones(p2,1) Xte];

W1 = rand(Q1,N+1);
W2 = rand(Q2,Q1+1);
W3 = rand(m-1,Q2+1);

MSETemp = zeros(1,T);

for i = 1:T
    V1 = W1*Xtr';
    Z1 = 1./(1+exp(-V1));
    S1 = [bias*ones(1,p);Z1];
    
    V2 = W2*S1;
    Z2 = 1./(1+exp(-V2));
    S2 = [bias*ones(1,p);Z2];
    
    G = W3*S2;
    Y = 1./(1+exp(-G));
    E = Ztr - Y';
    mse = (mean(mean(E.^2)));
    MSETemp(i) = mse;
    if(mse<MSETarget)
        MSE = MSETemp(1:i);
        return
    end
    
    df = Y.*(1-Y);
    dG3 = df.*E';
    DW3 = mu/N * dG3*S2';
    W3 = (1+a)*W3 + DW3;
    
    df = S2.*(1-S2);
    dG2 = df.*(W3' * dG3);
    dG2 = dG2(2:end,:);
    DW2 = mu/N * dG2*S1';
    W2 = (1+a)*W2 + DW2;
    
    df = S1.*(1-S1);
    dG1 = df.*(W2' * dG2);
    dG1 = dG1(2:end,:);
    DW1 = mu/N * dG1*Xtr;
    W1 = (1+a)*W1 + DW1;
    
end

    V1 = W1*Xte';
    Z1 = 1./(1+exp(-V1));
    S1 = [bias*ones(1,p2);Z1];
    
    V2 = W2*S1;
    Z2 = 1./(1+exp(-V2));
    S2 = [bias*ones(1,p2);Z2];
    
    G = W3*S2;
    Yp = 1./(1+exp(-G));
    Ypp = Yp';
for k = 1:size(Ypp,1)
    [~, pl(k)] = max(Ypp(k,:));
    [~, pa(k)] = max(Zte(k,:));
end
[cm,order] = confusionmat(pa,pl);
Accuracy = ((trace(cm))/sum(cm(:)))*100;
display(cm);
display(Accuracy);

    

