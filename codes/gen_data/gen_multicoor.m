function [c,cimg, img] = gen_multicoor(numer, radius)
%% Generate 'numer' number of (x,y) coordinates for non-overlapping dots with radius 'radius' 
%% Requires: 
%   gen_coor.m
img = zeros(227);
cimg = zeros(227); 
A = cell(numer,2);
[imX, imY] = meshgrid(1:227,1:227);

for i = 1:numer
    [center_XY,img] = gen_coor(img,radius);
    A{i} = center_XY;
    cimg(((imX-center_XY(1)).^2 + (imY-center_XY(2)).^2) < 1) = 1;
end

c = cell2mat(A);