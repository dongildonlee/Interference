import numpy as np
import pandas as pd

def anova2(actv,numer,area):
    ##
    # Input:
    #   1. actv: 4D activity matrix from DNN
    #   2. numer: numerosities
    #   3. area: areas
    # Output:
    #   Dataframe for a 2-way ANOVA
    ##
    categories = actv.shape[1]
    instances = actv.shape[2]
    actv_net2D = actv.reshape(actv.shape[0], categories*instances)
    # Find active units
    not_empty = np.any(actv_net2D,axis=1)
    inactv_units = [i for i,x in enumerate(not_empty) if not x]
    actv_units = np.setdiff1d(np.arange(43264),inactv_units)
    #
    df_stats = pd.DataFrame(index=np.arange(actv.shape[0]), columns = ['numer', 'area', 'inter','residual'])
    for i in actv_units:
        if i%1000 == 0:
            print("unit#:",i)
        df = pd.DataFrame({'numer': np.repeat(np.repeat(numer,len(numer)),instances),'area': np.tile(np.repeat(area,instances),len(area)),'activity':actv_net2D[i,:]})
        try:
            model = ols('activity ~ C(numer) + C(area) + C(numer):C(area)', data=df).fit()
            stat = sm.stats.anova_lm(model, typ=2)
            df_stats.loc[i,'numer'] = stat.loc['C(numer)', 'PR(>F)']
            df_stats.loc[i,'area'] = stat.loc['C(area)', 'PR(>F)']
            df_stats.loc[i,'inter'] = stat.loc['C(numer):C(area)', 'PR(>F)']
            df_stats.loc[i,'residual'] = stat.loc['Residual', 'PR(>F)']
        except:
            continue
    return df_stats


def gaussian_curve_fit(numbers, sizes, uoi, actv_net):
    ###
    ## INPUT:
    #   1. numbers: an array of numbers
    #   2. sizes: np array of sizes
    #   3. uoi: np array of neuron indices that are of our interest
    #   4. actv_net: raw activity data (e.g. 43264 x 100 x 100 shape)
    ## OUTPUT:
    #   Fitted parameters
    ###

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
