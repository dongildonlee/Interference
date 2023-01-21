load('/Users/dongillee/Documents/MATLAB/AlexNet/img_set_N2to20_S4to13_1000inst.mat');
img_set_f500 = img_set(:,:,:,1:500);

layer = 'relu4';
relu=4;

for network=1:10

    disp(network)
    for epochind=21:30:91
        epoch=epochind-1;
        disp(epoch)
        suffix = sprintf('_f500_network%d_relu%d_epoch%d.mat', network,relu,epoch);
        load(['/Users/dongillee/Documents/MATLAB/AlexNet/ILSVRC2012_AlexNet/HeGaussian_sig100_' num2str(network) '/network/net (' num2str(epoch) ').mat']);
        %net=nets{network};
        actv = get_activations3(img_set_f500, net, layer);
        save( ['actv', suffix], 'actv' );
    end 
end 