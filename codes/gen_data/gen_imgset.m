function [c,cimg, img_set] = gen_imgset(numbers, radii, inst)
%% Generate a 4D image stimulus set
%% Requires:
%   gen_multicoor.m
combn=length(numbers)*length(radii);
img_set = zeros(227,227,combn,inst);
max_radius=max(radii);

for n=1:length(numbers)
    for i=1:inst 
        [c,cimg, img] = gen_multicoor(numbers(n), max_radius);
        img_set(:,:,n*length(radii),i)=img;
        for r=1:length(radii)-1
            img=zeros(227);
            [imX, imY] = meshgrid(1:227,1:227);
            for co=1:length(c)
                img(((imX-c(co,1)).^2 + (imY-c(co,2)).^2) < radii(r)^2) = 1;
            end
            img_set(:,:,(n-1)*length(radii)+r,i)=img;
        end
    end
end


