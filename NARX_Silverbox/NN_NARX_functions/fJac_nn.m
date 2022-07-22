function J = fJac_nn(net,u)

% Compute the Jacobian of a single-hidden layer nn

if size(u,2) < size(u,1)
    u = u';
end

numInputs = size(net.IW,2);
numNodes = size(net.LW{2,1},2);

Wu = zeros(numNodes,numInputs); % input weights
bu = net.b{1};

for i=1:numInputs
    Wu(:,i) = net.IW{1,i};
end

Wy = net.LW{2,1}; % output layer weights
by = net.b{2};

z1 = Wu*u+bu;
a1 = tansig(z1);
z2 = Wy*a1+by;

J = zeros(1,numInputs,size(u,2));
for t=1:size(u,2)
J(1,:,t) = Wy*diag(dtansig(z1(:,t),a1(:,t)))*Wu;
end


end