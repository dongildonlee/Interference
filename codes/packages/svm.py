import numpy as np
import random

def gen_SVM_input(nidx, sidx, rep_per_ns_combo, img_inst):
    ###
    ## INPUT:
    #   1. nidx: index of numbers used for SVM training and test
    #   2. sidx: index of sizes used for SVM training and test
    #   3. rep_per_ns_combo: number of instances for each unique (number,size) combination
    #   4. img_inst: image instances available (e.g. 500 images for each combination)
    ## OUTPUT:
    #   Dataframe with examples of number, size and image indices for a pair of iamges being compared
    ###
    ncombs = np.array([[a, b] for idx, a in enumerate(nidx) for b in nidx[idx + 1:]]) # num1 cannot be equal to num2
    ncombs = np.concatenate([ncombs,ncombs[:,[1,0]]],axis=0)
    scombs = np.array([[a, b] for idx, a in enumerate(sidx) for b in sidx[:]]) # sz1 CAN be equal to sz2
    num_rows = len(ncombs)*len(scombs)*rep_per_ns_combo
    # Make a dataframe:
    pd_ns_idx = pd.DataFrame(index = np.arange(num_rows), columns = ['num1','num2','sz1','sz2','img1','img2'])
    pd_ns_idx.iloc[:,0:2] = np.repeat(ncombs, len(scombs)*rep_per_ns_combo,axis=0)
    pd_ns_idx.iloc[:,2:4] = np.tile(np.repeat(scombs, rep_per_ns_combo, axis=0), (len(ncombs),1))
    pd_ns_idx.loc[:,'img1'] = random.choices(np.arange(img_inst), k = num_rows)
    pd_ns_idx.loc[:,'img2'] = random.choices(np.arange(img_inst), k = num_rows)
    # Shuffle rows:
    pd_ns_idx = pd_ns_idx.sample(frac=1).reset_index(drop=True)
    return pd_ns_idx
