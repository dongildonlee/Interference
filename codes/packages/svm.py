import numpy as np
import pandas as pd
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

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
    d = df_idx
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


def SVM_fit(units, actv, exp):
    ###########################################################
    ## INPUT:
    #   1. units: unit IDs used for SVM training
    #   2. actv: activity matrix
    #   3. exp: experiment #
    ## OUTPUT:
    #   An array of prediction from the classifier
    ###########################################################
    # Get classifier:
    clf = make_pipeline(LinearSVC(random_state=1234, tol=1e-5, max_iter=1000000))
    scaler = StandardScaler()
    # Get training and test sets for experiment (exp):
    dir_path = os.path.dirname(os.path.realpath('../'))
    df_train = pd.read_csv(dir_path+'/dataframes/SVM/training_sets/training set idx for exp'+str(exp)+'.csv',index_col=0)
    df_test = pd.read_csv(dir_path+'/dataframes/SVM/test_sets/test set idx for exp'+str(exp) + '.csv', index_col=0)

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


def get_all_text_X(exps):
    ###########################################################
    ## INPUT:
    #   1. exps: SVM test trial #
    ## OUTPUT:
    #   SVM test trials with number size index in a dataframe
    ###########################################################

    dir_path = os.path.dirname(os.path.realpath('../'))
    all_test_X = pd.concat([pd.read_csv(dir_path+'/dataframes/SVM/test_sets/test set idx for exp'+str(exp) + '.csv').drop(columns=['Unnamed: 0']) for exp in exps])
    all_test_X.index = np.arange(all_test_X.shape[0])
    return all_test_X


def get_all_preds(net, relu, epoch, num_units, selectivity, exps):
    ###########################################################
    ## INPUT:
    #   1. net: network ID #
    #   2. relu: relu layer #
    #   3. epoch: training epoch #
    #   4. num_units: number of units used in SVM
    #   5. selevtivity: number, size or NS (number and size)
    #   6. exps: SVM test trials
    ## OUTPUT:
    #   predictions on the SVM test trials
    ###########################################################
    dir_path = os.path.dirname(os.path.realpath('../'))
    all_preds=pd.concat([pd.read_csv(dir_path+'/dataframes/SVM_predictions/SVM prediction of He untrained net'+str(net)+' relu'+str(relu)+' epoch'+str(epoch)+' '+str(num_units)+' '+str(selectivity)+' units that are randomly drawn from distribution exp' + str(exp)+ ' Jan2023.csv').drop(columns=['Unnamed: 0']) for exp in exps])
    all_preds.index = np.arange(all_preds.shape[0])
    return all_preds


def get_SVM_accuracy(pair_idx, all_test_X, all_preds):
    ###########################################################
    ## INPUT:
    #   1. pair_idx: SVM test trial #
    #   2. indices corresponding to congruent/incongruent trials
    #   3. all_test_X: test SVM trials
    #   4. all_preds: predictions on the SVM trials
    ## OUTPUT:
    #   SVM accuracy
    ###########################################################
    test_arr = ((all_test_X['num1'] > all_test_X['num2'])*2-1).to_numpy()[pair_idx]
    pred_arr = all_preds['0'].to_numpy()[pair_idx]
    accuracy = sum(test_arr == pred_arr)/len(test_arr)
    return accuracy


# def get_all_preds_temp(file):
#     all_preds=pd.concat([pd.read_csv(file+' exp' + str(exp)+ ' Dec12.csv').drop(columns=['Unnamed: 0']) for exp in np.arange(1,11)])
#     all_preds.index = np.arange(all_preds.shape[0])
#     return all_preds


def get_congruency(test_X, variable):
    ###########################################################
    ## INPUT:
    #   1. test_X: SVM test set in dataframe
    #   2. variable: currently only 'dot size' available
    ## OUTPUT:
    #   congruent and incongruent trial indices
    ###########################################################
    if variable == 'dot size':
        c1 = test_X.index.to_numpy()[(test_X['num1']<test_X['num2'])*(test_X['sz1']<test_X['sz2'])]
        c2 = test_X.index.to_numpy()[(test_X['num1']>test_X['num2'])*(test_X['sz1']>test_X['sz2'])]
        c = np.union1d(c1,c2)
        ic = np.setdiff1d(test_X.index.to_numpy(), c)

    return c, ic
