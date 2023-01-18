import pandas as pd

def units_for_svm(path, num_units):
    uoi = pd.read_csv(path+'/'+str(num_units)+' randomly sampled units from distribution of both units in He untrained net'+str(net)+' epoch'+str(epoch)+ ' relu'+str(relu)+'.csv').drop(columns=['Unnamed: 0'])['0'].to_numpy()
    return uoi
