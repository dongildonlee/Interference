%% Requires:
%   Have stimulus set named 'img_set' loaded
%   A cell named 'nets' (e.g. randomized AlexNet) containing multiple randomized deep networks loaded 

%% Parameters
layer = 'relu5';
relu = 5;
epoch = 0;
num_networks = 2;

%%
for network=1:num_networks
    disp(network)
    net = nets{network};
    %for epochind=21:30:91
        %epoch=epochind-1;
        %disp(epoch)
    suffix = sprintf('_f500_network%d_relu%d_epoch%d.mat', network,relu,epoch);
    %load(['/Users/dongillee/Documents/MATLAB/AlexNet/ILSVRC2012_AlexNet/HeGaussian_sig100_' num2str(network) '/network/net (' num2str(epoch) ').mat']);
        %net=nets{network};
    actv = get_activations(img_set, net, layer);
    save( ['test_actv', suffix], 'actv' );
    %end 
end 