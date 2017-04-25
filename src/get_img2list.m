function [] = get_img2list(varargin)
addpath('src');
opts.im_in_dir  = 'D:/data/img_align_celeba_crop';
opts.img_format = 'jpg';
opts.out_imdb_path = 'list_Celeb_full.mat';
opts.MAXELEMENTS = 400000;
opts.train_samples = 400000;
opts.train_valid_ratio = 1;

opts = vl_argparse(opts, varargin);

im_in_dir = dir(fullfile(opts.im_in_dir , ['*.', opts.img_format]));
opts.MAXELEMENTS = min(opts.MAXELEMENTS, numel(im_in_dir));

%% initialization
imdb = [];
imdb.opts = opts;
imdb.images.data = sort({im_in_dir.name});
imdb.images.set = ones(1, opts.MAXELEMENTS, 'single');

for i = 1:numel(imdb.images.data)
    imdb.images.data{i} = [opts.im_in_dir, '/', imdb.images.data{i}];
end 

imdb.meta.sets = {'train','val','test'};

%% Cut data length
imdb.images.data   =  imdb.images.data(1:opts.MAXELEMENTS);
imdb.images.set = imdb.images.set(1:opts.MAXELEMENTS);
count = opts.MAXELEMENTS*opts.train_valid_ratio;
imdb.images.set(count:end) = 2;

save(opts.out_imdb_path, 'imdb');
fprintf('get_img2list is complete\n') ;

return;


