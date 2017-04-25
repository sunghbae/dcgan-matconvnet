function [ net ] = get_BN_dag(net, i, input, output, param)
%GET_CONV_DAG Summary of this function goes here
%   Detailed explanation goes here

i = i+1;  net.addLayer(sprintf('bn%d',i),   dagnn.BatchNorm('numChannels', param.depth, 'epsilon', param.epsilon), input,  output,{sprintf('gg%d',i), sprintf('bb%d',i), sprintf('mm%d',i)});

return;