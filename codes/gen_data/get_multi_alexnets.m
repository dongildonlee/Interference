%% Make multiple randomized AlexNets:
num_nets_tomake = 20;

nets = cell(num_nets_tomake,1);

for i = 1:num_nets_tomake
    net = alexnet;
    layers_to_randomize = [2,6,10,12,14];
    [random_net, lim, random_Weights, random_Biases] = Initializeweight_he2(net, layers_to_randomize, 2, 1);
    nets{i} = random_net;
end