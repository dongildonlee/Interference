import numpy as np
import random

def gen_SVM_input(nidx, sidx, rep_per_ns_combo, img_inst):
    ###########################################################
    ## INPUT:
    #   1. nidx: index of numbers used for SVM training and test
    #   2. sidx: index of sizes used for SVM training and test
    #   3. rep_per_ns_combo: number of instances for each unique (number,size) combination
    #   4. img_inst: image instances available (e.g. 500 images for each combination)
    ## OUTPUT:
    #   Dataframe with examples of number, size and image indices for a pair of iamges being compared
    ###########################################################
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


def get_SVM_actv(df_idx, units, actv):
    ###########################################################
    ## INPUT:
    #   1. df_idx: Dataframe with examples of number, size and image indices for a pair of iamges being compared
    #   2. units: unit IDs used in SVM
    #   3. actv: activity matrix
    ## OUTPUT:
    #   1. X: activity corresponding to the input dataframe
    ###########################################################
    d = SVM_idx
    actv_list=[]
    for im in d.index:
        #print(im)
        actv_left = actv[units, d.loc[im,'num1'], d.loc[im,'sz1'], d.loc[im,'img1']]
        actv_right = actv[units, d.loc[im,'num2'], d.loc[im,'sz2'], d.loc[im,'img2']]
        actv_ = np.concatenate((actv_left, actv_right))
        actv_list.append(actv_)
    X=np.vstack(actv_list)
    return X


def get_y(df_idx):
    ###########################################################
    ## INPUT:
    #   1. df_idx: Dataframe with examples of number, size and image indices for a pair of iamges being compared
    ## OUTPUT:
    #   1. y: An array of correct responses
    ###########################################################
    is_left_larger = df_idx.loc[:,'num1'] > df_idx.loc[:,'num2']
    y = is_left_larger*2-1
    return y


def SVM_fit2(df_train, df_test, units, actv):
    ###########################################################
    ## INPUT:
    #   1. units: unit IDs used for SVM training
    #   2. actv: activity matrix
    ## OUTPUT:
    #   An array of prediction from the classifier
    ###########################################################
    clf = make_pipeline(LinearSVC(random_state=1234, tol=1e-5, max_iter=1000000))
    scaler = StandardScaler()

    # Process training data and fit classifier:
    X_tr = get_SVM_actv(df_train, units, actv)
    X_tr_scld = scaler.fit_transform(X_tr)
    y_tr = get_y(df_train)
    clf.fit(X_tr_scld, y_tr)

    # Process test data and get prediction:
    X_tst = get_SVM_actv(df_test, units, actv)
    X_tst_scld = scaler.transform(X_tst)
    y_test = get_y(df_test)
    y_pred = np.array([clf.predict([X_tst_scld[i,:]])[0] for i in np.arange(X_tst_scld.shape[0])])

    return y_pred
