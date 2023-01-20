for net in np.arange(9,11):
    print("net:", net)
    for epoch in np.array([10,20,40,50,70,80]):
        print("epoch:", epoch)
        #f500 = h5py.File('RESEARCH (updated June 23)/raw response/He/Untrained/actv_f500_network'+str(net)+'.mat','r')
        f500 = h5py.File('raw response/He/relu'+str(relu)+'/Epochs/actv_f500_network'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+'.mat', 'r')
        actv_=f500['actv'][:]
        actv = np.transpose(actv_, (2,1,0))
        #actv_2D = actv.reshape(actv.shape[0], actv.shape[1]*actv.shape[2])

        # Take activity corresponding to size 7 to 13:
        take = np.arange(0,100).reshape(10,10)[:,min_sz_idx:max_sz_idx+1].reshape(len(numbers)*(max_sz_idx-min_sz_idx+1))
        actv_szAtoB = actv[:,take,:]
        actv_2D = actv_szAtoB.reshape(actv_szAtoB.shape[0], actv_szAtoB.shape[1]*actv_szAtoB.shape[2])
        df_anova2 = pd.DataFrame(index=units, columns = ['number', 'size', 'inter', 'residual'])
        ## Perform 2-way ANOVA with parallel computing:


        for tt in np.arange(num_blocks):
            try:
                #unt = np.arange(tt*1352, (tt+1)*1352)
                unt = np.arange(tt*4056, (tt+1)*4056)
                start_time = time.time()
                stats = Parallel(n_jobs=-1)(delayed(ffns.anova2_single)(actv_2D, un, numbers, sizes, inst) for un in unt)
                print("--- %s seconds ---" % (time.time() - start_time))

                ## Save the data:
                #stats_pval = pd.concat(stats).iloc[:,3].to_numpy().reshape(1352,4)
                stats_pval = pd.concat(stats).iloc[:,3].to_numpy().reshape(4056,4)
                df_anova2.iloc[unt,:]=stats_pval
                df_anova2.to_csv('ANOVA2 results/He/Relu'+str(relu)+'/size7to13/df_anova2 for He initialized net'+str(net)+'_relu'+str(relu)+'_epoch'+str(epoch)+' size 7to13 500inst.csv')
            except:
                continue
