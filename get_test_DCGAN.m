function [ net ] = get_test_DCGAN(varargin)
addpath('src');

% cpu and gpu settings 
opts.idx_gpus = 1; % 0: cpu
opts.num_images = 32; % images to be generated
opts.matconvnet_path = 'matconvnet-1.0-beta24/matlab/vl_setupnn.m';
opts.net_path = 'net/net_dcgan.mat'; 
opts.save_img_path = 'test_img_DCGAN';

opts = vl_argparse(opts, varargin);

if ~exist(opts.save_img_path, 'dir'), mkdir(opts.save_img_path) ; end
run(opts.matconvnet_path);

%% load network
net = load(opts.net_path);
net = net.net(1); % idx 1: Generator, 2: Discriminator
net = dagnn.DagNN.loadobj(net);
net.mode = 'test';

if opts.idx_gpus >0
    gpuDevice()
    net.move('gpu');
end

rng('shuffle') ;% seeding random number generator

for i = 1:opts.num_images 
    tic;
    c= clock;
    input = single(randn(1, 1, 100, 64, 'single'));
    if opts.idx_gpus >0,   input = gpuArray(input);    end
    
    net.eval({'data', input}) ;
    im_out = gather(net.vars(end).value);
    im_out = get_tensor2img(im_out);
    im_out = uint8((im_out+1)/2*255); % single [-1:1] to uint8 [0:255]

    imwrite(im_out, sprintf('%s/%d_%d_%d_%d_%d_%1.f.png',  opts.save_img_path, c(1), c(2), c(3), c(4), c(5), c(6)*1000));
   
    fprintf('%d images processed (%.3f Hz)\n', i, toc)
    
end
fprintf('get_test_DCGAN is complete\n');

return;

function [im_out] = get_tensor2img(tensor_in)
f = size(tensor_in);
num_im_height = floor(sqrt(f(4)));
im_out = zeros(f(1).*num_im_height, f(2).*num_im_height, f(3), 'single');

cnt = 0;
for y = 1:num_im_height
    for x = 1:num_im_height
        cnt = cnt + 1;
        idx_y = (y-1)*f(1)+1:y*f(1);
        idx_x = (x-1)*f(2)+1:x*f(2);
        im_out(idx_y,idx_x,:) = tensor_in(:,:,:,cnt);
        
    end
end

return;
