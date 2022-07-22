function yhat = fPredict_NN_singleHidden(net,x)
W1 = zeros(size(net.IW{1,1},1),size(x,1));
for i=1:size(x,1)
W1(:,i) = net.IW{1,i};
end
W2 = net.LW{2,1};
b1 = net.b{1};
b2 = net.b{2};
a1 = tansig(W1*x+b1);
yhat = W2*a1+b2;
end