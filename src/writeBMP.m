function [  ] = writeBMP(filePath, singleIm1 )
%WRITEGIF Summary of this function goes here
%   Detailed explanation goes here

singleIm1(singleIm1<0) = 0;
singleIm1(singleIm1>1) = 1;
im1 = uint8(255*singleIm1);

imwrite(im1,filePath);


return;
