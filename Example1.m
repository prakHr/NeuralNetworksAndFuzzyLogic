%Error using randperm
%K must be less than or equal to N.

%table1 = xlsread('data.xlsx');
%size(table1);%ans = 349 3
%p = randperm(size(table1,1),50);% gets random data points of 50 cols between 1 to 349


%table=[1 2 3;3 2 1];
%size(table,1);%(2 3)=>2
%X=ones(size(table,1),1);
%X;%[1;1]
%X=[X table];%[1 1 2 3;1 3 2 1]

%a=[12 13 14];
%a(a>13)=14.5;

%b=a(a>13);
%b;
%greaterThanThree=a>3;
%greaterThanThree;


%X = rand(5,3);
%n=numel(X);
%n

%plot(X,'ro-','LineWidth',3);
%legend('rand1','rand2','rand3');

%disp('     Corn      Oats      Hay');
%disp(X);%disp(X) displays the value of variable X without printing the variable name

%X = '<a href = "https://www.mathworks.com">MathWorks Web Site</a>';
%disp(X);

%name = 'Alice';   
%age = 12;
%X = [name,' will be ',num2str(age),' this year.'];
%disp(X);

%name = 'Alice';   
%age = 12;
%X = [name,' will be ',num2str(age),' this year.'];
%disp(X);

%name = 'Alice';   
%age = 12;
%fprintf('%s will be %d this year.\n',name,age);