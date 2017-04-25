function [  ] = writeGIF(filePath, singleIm1, singleIm2, delayTime )
%WRITEGIF Summary of this function goes here
%   Detailed explanation goes here
% h = [0 -1 0; -1 4 -1; 0 -1 0];
% singleIm1 = Laplacian_u(singleIm1, h); 
% singleIm1 = singleIm1 + 0.5;

singleIm1 = gather(singleIm1);
singleIm2 = gather(singleIm2);

singleIm1(singleIm1<0) = 0;
singleIm1(singleIm1>1) = 1;
im1 = uint8(255*singleIm1);



[A,map] = rgb2ind(im1,256); 
imwrite(A,map,filePath,'gif','LoopCount',Inf,'DelayTime',delayTime);

% singleIm2 = singleIm2 + 0.5;
singleIm2(singleIm2<0) = 0;
singleIm2(singleIm2>1) = 1;
im2 = uint8(255*singleIm2);


[A,map] = rgb2ind(im2,256);
imwrite(A,map,filePath, 'gif', 'WriteMode', 'append', 'DelayTime',delayTime);


return;
