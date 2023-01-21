function actv_tr = get_activations(image_set,net,layer)

% parameters to be used

num_img = prod(size(image_set,[3,4]));

% image-set transformation:

img_set_tr = reshape(image_set, [227,227,1,num_img]); 
img_set_cat = 255 * cat(3, img_set_tr, img_set_tr, img_set_tr); % converting to 0-255 scale
actv = activations(net,img_set_cat,layer);
actv_tr = reshape(actv, [size(actv,1)*size(actv,2)*size(actv,3),size(image_set,3),size(image_set,4)]);

end