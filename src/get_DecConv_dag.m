function [ net ] = get_DecConv_dag(net, i, input, output, param)
%GET_DECCONV_DAG Summary of this function goes here
%   Detailed explanation goes here

f = param.f;
f(3) = param.f(4);
f(4) = param.f(3);

pad = param.pad;
stride = param.stride;
bias = param.bias;
if bias == true
    net.addLayer(sprintf('deconv%d',i), dagnn.ConvTranspose('size',f, 'upsample',stride, 'crop',pad, 'hasBias', bias),input, output, {sprintf('w%d',i),sprintf('b%d',i)});
else
    net.addLayer(sprintf('deconv%d',i), dagnn.ConvTranspose('size',f, 'upsample',stride, 'crop',pad, 'hasBias', bias),input, output, {sprintf('w%d',i)});
end    


end

