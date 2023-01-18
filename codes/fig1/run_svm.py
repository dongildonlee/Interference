import numpy as np
import pandas as pd
import sys
sys.path.append('../')
from packages import actv_analysis, svm
#from packages import svm

relu=5
epochs=np.arange(0,91,10)
exps = np.arange(1,11)
min_sz_idx=0; max_sz_idx=9
selectivity =['number','size','both'][2]

num_samples = 100
rate_threshold = 0.05

data_from = 'dataframes/SVM/units/'+str(num_samples)+' units sampled from distribution higher than '+str(rate_threshold)+' response rate including PN2 and PN20/'

folder='dataframes/SVM'

save_folder= 'dataframes/SVM/Relu'+str(relu)+'/'+str(num_samples)+' units sampled from distribution higher than '+str(rate_threshold)+' response rate including PN2 and PN20/'

for epoch in epochs:
    print("epoch:",epoch)
    for net in np.arange(1,3):
        print("net:",net)
        actv_net = actv_analysis.get_actv_net(net=net,relu=relu,epoch=epoch)
        actv = actv_net.reshape(43264,10,10,500)
        uoi = pd.read_csv(data_from+str(num_samples)+' randomly sampled units from distribution of both units in He untrained net'+str(net)+' epoch'+str(epoch)+ ' relu'+str(relu)+'.csv').drop(columns=['Unnamed: 0'])['0'].to_numpy()

        #start_time = time.time()
        y_preds = Parallel(n_jobs=-1)(delayed(SVM_fit2)(folder, net, relu, epoch, exp, uoi, actv) for exp in exps)

        #print("--- %s seconds ---" % (time.time() - start_time))
        for exp in exps:
            pd.Series(y_preds[exp-1]).to_csv(save_folder+'SVM prediction of He untrained net'+str(net)+' relu'+str(relu)+' epoch'+str(epoch)+' '+str(num_samples)+' '+str(selectivity)+' units that are randomly drawn from distribution exp' + str(exp)+ ' Jan2023.csv', index=True)
