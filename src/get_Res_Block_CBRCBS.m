function [net, i] = get_Res_Block_CBRCBS( net, i, conv_param, bn_param,relu_param)
%GET_BLOCK_CBR Summary of this function goes here
%   Detailed explanation goes here

i = i+1;  net = get_Conv_dag(net, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
i = i+1;  net.addLayer(sprintf('bn%d',i),   dagnn.BatchNorm('numChannels', bn_param.depth, 'epsilon', bn_param.epsilon), sprintf('x%d',i-1),  sprintf('x%d',i),{sprintf('g%d',i), sprintf('b%d',i), sprintf('m%d',i)});
i = i+1;  net = get_ReLU_dag(   net, i, sprintf('x%d',i-1), sprintf('x%d',i), relu_param);

i = i+1;  net = get_Conv_dag(net, i, sprintf('x%d',i-1), sprintf('x%d',i), conv_param);
i = i+1;  net.addLayer(sprintf('bn%d',i),   dagnn.BatchNorm('numChannels', bn_param.depth, 'epsilon', bn_param.epsilon), sprintf('x%d',i-1),  sprintf('x%d',i),{sprintf('g%d',i), sprintf('b%d',i), sprintf('m%d',i)});
i = i+1;  net.addLayer(sprintf('sum%d',i), dagnn.Sum(),{sprintf('x%d',i-6), sprintf('x%d',i-1)}, sprintf('x%d',i));

end

