function [net, i] = get_Conv_BN_ReLU_dag(net, input, i, conv_param, bn_param, relu_param)
%GET_CONVRELU Summary of this function goes here
%   Detailed explanation goes here
% conv_param.{f, pad, stride, bias}
    
i = i+1;    net.addLayer(sprintf('conv%d',i),   dagnn.Conv('size',conv_param.f,'pad',conv_param.pad,'stride',conv_param.stride,'hasBias',conv_param.bias),input,sprintf('x%d',i), {sprintf('w%d',i),sprintf('b%d',i)} );

i = i+1;    net.addLayer(sprintf('bn%d',i),   dagnn.BatchNorm(), sprintf('x%d',i),  sprintf('x%d',i),{sprintf('g%d',i), sprintf('b%d',i), sprintf('m%d',i)});
%     pidx = net.getParamIndex({sprintf('g%d',i), sprintf('b%d',i), sprintf('m%d',i)});
%     net.params(pidx(1)).weightDecay = 0;
%     net.params(pidx(2)).weightDecay = 0; 
%     net.params(pidx(3)).learningRate = 0.1;
%     net.params(pidx(3)).trainMethod = 'average'; 
i = i+1;    net.addLayer(sprintf('relu%d',i),   dagnn.ReLU('leak',relu_param), sprintf('x%d',i-1), sprintf('x%d',i));

return;