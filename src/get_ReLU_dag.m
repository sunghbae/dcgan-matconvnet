function [ net ]  = get_ReLU_dag(net, i, input, output, param)
%GET_RELU Summary of this function goes here
%   Detailed explanation goes here
net.addLayer(sprintf('Relu%d',i),   dagnn.ReLU('leak', param), input, output);
 
end

