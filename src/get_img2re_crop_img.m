function [] = get_img2re_crop_img(varargin)

opts.im_in_dir  = '/data/img_align_celeba';
opts.im_out_dir = '/data/img_align_celeba_crop';
opts.img_format = 'jpg';
opts.jpg_quality = 100;
opts.scale_factor = 1/2;
opts.crop_start = [25 13];
opts.crop_size = [64 64];

opts = vl_argparse(opts, varargin);

if ~exist(opts.im_out_dir , 'dir'), mkdir(opts.im_out_dir) ; end

im_in_dir = dir(fullfile(opts.im_in_dir , ['*.', opts.img_format]));

for i = 1:1:numel(im_in_dir)
    tic
    im_tmp = imread(fullfile(opts.im_in_dir, im_in_dir(i).name));
    im_tmp = imresize(im_tmp, opts.scale_factor, 'bicubic');
    
    im_tmp = im_tmp(opts.crop_start(1):opts.crop_start(1)+opts.crop_size(1)-1, opts.crop_start(2):opts.crop_start(2)+opts.crop_size(2)-1,:);
    
    imwrite(im_tmp, fullfile(opts.im_out_dir, im_in_dir(i).name), 'Quality', opts.jpg_quality);
    if mod(i,1000)==0
        fprintf('%d images processed (%.3f Hz)\n', i, 1/toc)
    end
end

fprintf('get_img2re_crop_img is complete\n') ;

return;


