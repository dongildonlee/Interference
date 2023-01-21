import pandas as pd

def units_for_svm(path, num_units, net, epoch, relu):
     ###########################################################
    ## INPUT:
    #   1. path: path to the csv file
    #   2. num_units: number of units used in SVM
    #   3. net: network ID #
    #   4. epoch: training epoch #
    #   5. relu: relu layer #
    ## OUTPUT:
    #   an array of unit IDs
    ###########################################################
    uoi = pd.read_csv(path+'/'+str(num_units)+' randomly sampled units from distribution of both units in He untrained net'+str(net)+' epoch'+str(epoch)+ ' relu'+str(relu)+'.csv').drop(columns=['Unnamed: 0'])['0'].to_numpy()
    return uoi
