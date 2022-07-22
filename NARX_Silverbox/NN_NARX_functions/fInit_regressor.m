function net = fInit_regressor(numInputs,numNodes,nIter)

% Single hidden layer neural network for regression
numLayers = 2;

biasConnect = ones(numLayers,1);
inputConnect = [ones(1,numInputs);
                zeros(1,numInputs)];           

layerConnect = zeros(numLayers,numLayers);

layerConnect(2,1) = 1; % layer two connects to layer one

outputConnect = [zeros(1,numLayers-1) 1];

net = network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);

% set the activation functions of the layers
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'purelin'; 


% set delay in the feedback loop
% feedforward

% set number of neurons per layer
net.layers{1}.size = numNodes;
net.layers{2}.size = 1;

% set training setting - levenberg marquardt
net.trainFcn = 'trainlm';
net.adaptFcn = 'adaptwb';
net.performFcn= 'mse';
net.plotFcns = {'plotresponse','plotperform'};
net.trainParam.epochs = nIter;
net.trainParam.min_grad = 1e-15;
net.trainParam.max_fail = 10;
net.trainParam.mu_dec = 0.5;
net.trainParam.mu_inc = 2;
net.trainParam.mu_max = 1e15;

net.LW{2,1} = rands(1,numNodes);
for i=1:numInputs
net.IW{1,i} = rands(numNodes,0);
end
net.b{1} = rands(numNodes);
net.b{2} = rands(1,1);
%view(net)
end