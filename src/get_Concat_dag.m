function [ net ] = get_Concat_dag(net, i, input, output)
%GET_DECCONV_DAG Summary of this function goes here
%   Detailed explanation goes here
net.addLayer(sprintf('concat%d',i), dagnn.Concat(),input, output);

end

