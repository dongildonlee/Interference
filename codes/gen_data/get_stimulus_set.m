%% Generate an image set corresponding to the set parameters:

%% Parameters
numbers = [2:2:20];
radii = [4:13];
instances = 100;

%% Generate an image set
[c,cimg, img_set] = gen_imgset(numbers, radii, instances);

%% Save data
save('stimulus_set_2023.mat','img_set');