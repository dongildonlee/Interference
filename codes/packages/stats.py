import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

# def anova2(actv,numer,area):
#     ##
#     # Input:
#     #   1. actv: 4D activity matrix from DNN
#     #   2. numer: numerosities
#     #   3. area: areas
#     # Output:
#     #   Dataframe for a 2-way ANOVA
#     ##
#     categories = actv.shape[1]
#     instances = actv.shape[2]
#     actv_net2D = actv.reshape(actv.shape[0], categories*instances)
#     # Find active units
#     not_empty = np.any(actv_net2D,axis=1)
#     inactv_units = [i for i,x in enumerate(not_empty) if not x]
#     actv_units = np.setdiff1d(np.arange(43264),inactv_units)
#     #
#     df_stats = pd.DataFrame(index=np.arange(actv.shape[0]), columns = ['numer', 'area', 'inter','residual'])
#     for i in actv_units:
#         if i%1000 == 0:
#             print("unit#:",i)
#         df = pd.DataFrame({'numer': np.repeat(np.repeat(numer,len(numer)),instances),'area': np.tile(np.repeat(area,instances),len(area)),'activity':actv_net2D[i,:]})
#         try:
#             model = ols('activity ~ C(numer) + C(area) + C(numer):C(area)', data=df).fit()
#             stat = sm.stats.anova_lm(model, typ=2)
#             df_stats.loc[i,'numer'] = stat.loc['C(numer)', 'PR(>F)']
#             df_stats.loc[i,'area'] = stat.loc['C(area)', 'PR(>F)']
#             df_stats.loc[i,'inter'] = stat.loc['C(numer):C(area)', 'PR(>F)']
#             df_stats.loc[i,'residual'] = stat.loc['Residual', 'PR(>F)']
#         except:
#             continue
#     return df_stats


def anova2_single(actv_2D, unit, numbers, sizes, instances):
    ########################################################
    # Input:
    #   1. act_2D: 2D activity matrix from DNN
    #   2. unit: unit whose activities will be analyzied
    #   3. numbers
    #   4. sizes
    #   5. instances: number of instances per each (number,size) combnination
    # Output:
    #   anova2 result for the unit
    ########################################################
    df = pd.DataFrame({'number': np.repeat(np.repeat(numbers,len(sizes)),instances),'size': np.tile(np.repeat(sizes,instances),len(numbers)),'activity':actv_2D[unit,:]})
    model = ols('activity ~ C(number) + C(size) + C(number):C(size)', data=df).fit()
    stat = sm.stats.anova_lm(model, typ=2)
    return stat


def get_selectivity(df_anova2):
    ########################################################
    # Input:
    #   1. df_anova2: dataframe with anova2 (2-way ANOVA) results
    # Output:
    #   a dataframe with selectivity for number, size and both
    ########################################################
    df_selectivity = pd.DataFrame(index=df_anova2.index.to_numpy(), columns=['selectivity'])
    unit_noresponse = df_anova2.index[np.sum(df_anova2.isnull().iloc[:,0:3],axis=1)==3].to_numpy()
    df_anova2 = df_anova2.drop(labels=unit_noresponse)
    number_selective = df_anova2.index[df_anova2.loc[:, 'number'] < 0.01].to_numpy()
    size_selective = df_anova2.index[df_anova2.loc[:, 'size'] < 0.01].to_numpy()
    interaction = df_anova2.index[df_anova2.loc[:, 'inter'] < 0.01].to_numpy()
    # Populate the dataframe with selectivity:
    df_selectivity.loc[np.setdiff1d(np.setdiff1d(number_selective, size_selective), interaction), 'selectivity'] = 'number'
    df_selectivity.loc[np.setdiff1d(np.setdiff1d(size_selective, number_selective), interaction), 'selectivity'] = 'size'
    NS_units = np.intersect1d(number_selective, size_selective)
    df_selectivity.loc[np.setdiff1d(NS_units, interaction), 'selectivity'] = 'NS NI'
    df_selectivity.loc[np.intersect1d(NS_units, interaction), 'selectivity'] = 'NS I'
    return df_selectivity



def gaussian_curve_fit(numbers, sizes, uoi, actv_net):
    ########################################################
    ## INPUT:
    #   1. numbers: an array of numbers
    #   2. sizes: np array of sizes
    #   3. uoi: np array of neuron indices that are of our interest
    #   4. actv_net: raw activity data (e.g. 43264 x 100 x 100 shape)
    ## OUTPUT:
    #   Fitted parameters
    ########################################################

    # average activity data across instances (e.g.n=100) and reshape the structure to (# of neurons)x(# of numbers)x(# of sizes)
    x = np.log2(numbers)
    avg_actv_10x10 = np.mean(actv_net,axis=2).reshape(actv_net.shape[0],len(numbers),len(sizes))
    PNidx4each_size = np.argmax(avg_actv_10x10, axis=1)
    PN4each_size = numbers[PNidx4each_size]
    df_pn = pd.DataFrame(index=np.arange(43264), columns = sizes, data = PN4each_size)

    popts_sz = []

    for s in np.arange(len(sizes)):
        print(sizes[s])

        popts2 = pd.DataFrame(index = np.arange(43264), columns = ['a','x0','sigma','pcov','r2'])
        avg_actv = avg_actv_10x10[:,:,s]
        avg_actv_norm = normed_data(avg_actv)

        for i in uoi:
            if np.mod(i,1000)==0:
                print("size:", s, " unit:",i)
            try:
                y = avg_actv_norm[i,:]
                # weighted arithmetic mean (corrected - check the section below),
                #mean = sum(x*y) / sum(y)\n",
                mean = np.log2(df_pn.loc[i,sizes[s]])
                #sigma = np.sqrt(sum(y*(x-mean)**2) / sum(y))\n",
                sigma=1
                popt2,pcov2 = sp.optimize.curve_fit(gaus, x, y, p0=[1,mean,sigma])
                y_pred = gaus(x,*popt2)
                r2 = r2_score(y,y_pred)
                #index.append(i)
                #ls_popt.append(popt)\n",
                popts2.iloc[i,0:3] = popt2
                popts2.loc[i,'pcov'] = pcov2
                popts2.loc[i,'r2'] = r2
            except:
                continue
        popts_sz.append(popts2)
    return popts_sz
