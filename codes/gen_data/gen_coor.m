function [center_XY,image] = gen_coor(image, radius)
%% Generate center coordinate of a dot stimulus and an image.
% image: image in which a dot will be added
    
    max_scale=1;
    max_bound = floor(227/max_scale); 
    min_bound = 227-max_bound;
    dot_added = 0;
    [imX, imY] = meshgrid(1:227,1:227);

    while dot_added ~= 1
        center_XY = randi([min_bound+radius+1, max_bound-radius-1],[1,2]);

        if all(image(((imX - center_XY(1)).^2 + (imY-center_XY(2)).^2) < radius^2) == 0)
        
           image(((imX-center_XY(1)).^2 + (imY-center_XY(2)).^2) < radius^2) = 1;
           dot_added=1;
        else
           continue;
        end
    end
end