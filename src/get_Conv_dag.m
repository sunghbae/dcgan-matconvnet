function [ net ] = get_Conv_dag(net, i, input, output, param)
%GET_CONV_DAG Summary of this function goes here
%   Detailed explanation goes here
f = param.f;
pad = param.pad;
stride = param.stride;
bias = param.bias;

% padUpDown = (f(1)-1)/2;
% padLeftRight = (f(2)-1)/2;
if( bias == true)
    net.addLayer(sprintf('conv%d',i),   dagnn.Conv('size',f,'pad',pad,'stride',stride,'hasBias',bias), input, output,  {sprintf('w%d',i),sprintf('b%d',i)});   
else
    net.addLayer(sprintf('conv%d',i),   dagnn.Conv('size',f,'pad',pad,'stride',stride,'hasBias',bias), input, output,  {sprintf('w%d',i)});   
end

% pidx = net.getParamIndex({sprintf('w%d',i),sprintf('b%d',i)});
% net.params(pidx(1)).trainMethod = 'adam'; 
% net.params(pidx(2)).trainMethod = 'adam'; 

return;