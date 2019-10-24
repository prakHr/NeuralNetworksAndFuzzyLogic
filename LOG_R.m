function yp = LOG_R(n,T,alpha,xtr,xte,ymtr)
%function does logistic regression with inputs as num of
%iterations,instances,learning rate
    w=rand(n+1,1);
    for t=1:T
        h=logsig(xtr*w);
    
        for j=1:n+1
            w(j)=w(j)-alpha*((h-ymtr)'*xtr(:,j));
        end
        J(t)=0.5*(norm(h-ymtr)^2);
        wn(:,t)=w;
    
    end
    figure;
    subplot(2,1,1);
    plot(1:T,J);
    subplot(2,1,2);
    plot3(wn(2,:),wn(3,:),J);

    yp=1./(1+exp(-(xte*w)));
    yp=(yp>0.5);
end

