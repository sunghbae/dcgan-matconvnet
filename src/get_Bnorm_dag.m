function net = get_Bnorm_dag(net, i, nCh, input, output )
%GET_BNORM_DAG Summary of this function goes here
%   Detailed explanation goes here

net.addLayer(sprintf('bn%d',i),   dagnn.BatchNorm('numChannels',nCh), input,  output,{sprintf('g%d',i), sprintf('b%d',i), sprintf('m%d',i)});
pidx = net.getParamIndex({sprintf('g%d',i), sprintf('b%d',i), sprintf('m%d',i)});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0; 
net.params(pidx(3)).learningRate = 0.1;
net.params(pidx(3)).trainMethod = 'average'; 

end

