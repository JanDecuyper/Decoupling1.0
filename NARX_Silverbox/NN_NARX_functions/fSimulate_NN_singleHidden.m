function y = fSimulate_NN_singleHidden(net,nu,ny,u)

W1 = zeros(size(net.IW{1,1},1),size(u,1));
for i=1:size(u,1)
    W1(:,i) = net.IW{1,i};
end
W2 = net.LW{2,1};
b1 = net.b{1};
b2 = net.b{2};

x = zeros(nu+ny,1);
x(1:nu) = u(1:nu,1);
y = zeros(1,size(u,2));
for t=1:size(u,2)-1
    a1 = tansig(W1*x+b1);
    y(t) = W2*a1+b2;
    
    x(1:nu) = u(1:nu,t+1);
    x(nu+2:nu+ny) = x(nu+1:nu+ny-1);
    x(nu+1) = y(t);
end

end


