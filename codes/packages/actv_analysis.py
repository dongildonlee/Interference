import numpy as np
import os
import h5py
import time

def get_actv_net(net,relu,epoch):
    ###
    ## INPUT:
    #   1. net: network ID
    #   2. epoch: training epoch
    #   3. relu: relu layer
    ## OUTPUT:
    #   Activity matrix corresponding to the input parameters
    ###
    start_time = time.time()
    dir_path = os.path.dirname(os.path.realpath('../'))
    print(dir_path)
    Unt_f500 = h5py.File(dir_path+'/data/raw_response/actv_f500_network'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+'.mat', 'r')
    actv_ = Unt_f500['actv'][:]
    actv = np.transpose(actv_, (2,1,0))

    print("--- %s seconds ---" % (time.time() - start_time))
    return actv


def get_PNs(actv, numbers, sizes, min_sz_idx, max_sz_idx):
    ###
    ## INPUT:
    #   1. actv: activity matrix
    #   2. numbers
    #   3. sizes
    #   4. min_sz_idx = index corresponding to the minimum size
    #   5. max_sz_idx = index corresponding to the maximum size
    ## OUTPUT:
    #   preferred numerosity by size, overall preferred numerosity
    ###
    avg_actv = np.nanmean(actv, axis=2)
    avg_actv_nxs_ = avg_actv.reshape(actv.shape[0], len(numbers), len(sizes))
    avg_actv_nxs = avg_actv_nxs_[:,:,min_sz_idx:max_sz_idx+1]
    PN_by_size = numbers[np.argmax(avg_actv_nxs, axis=1)]
    oPN = numbers[np.argmax(np.mean(avg_actv_nxs,axis=2),axis=1)]
    return PN_by_size, oPN


def get_PSs(actv, numbers, sizes, min_sz_idx, max_sz_idx):
    ###
    ## INPUT:
    #   1. actv: activity matrix
    #   2. numbers
    #   3. sizes
    #   4. min_sz_idx = index corresponding to the minimum size
    #   5. max_sz_idx = index corresponding to the maximum size
    ## OUTPUT:
    #   preferred numerosity by number, overall preferred size
    ###
    avg_actv = np.nanmean(actv, axis=2)
    avg_actv_nxs_ = avg_actv.reshape(actv.shape[0], len(numbers), len(sizes))
    avg_actv_nxs = avg_actv_nxs_[:,:,min_sz_idx:max_sz_idx+1]
    PS_by_num = sizes[min_sz_idx:max_sz_idx+1][np.argmax(avg_actv_nxs, axis=2)]
    oPS = sizes[min_sz_idx:max_sz_idx+1][np.argmax(np.mean(avg_actv_nxs,axis=1),axis=1)]
    return PS_by_size, oPS
