def runTileAdjacentMLConstruction(objid,workerErrorfunc,Qjfunc,A_percentile,Niter=10,DEBUG=False,PLOT_LIKELIHOOD=False,PLOT=False):
#     ##########
#     PLOT=True
#     DATA_DIR="output_15"
#     objid=3
#     workerErrorfunc="GTLSA"
#     Qjfunc=QjGTLSA
#     A_percentile=90
#     Niter=100
#     DEBUG=True
#     PLOT_LIKELIHOOD=False
#     ##########

    # ML Construction with T init as high confidence tiles 
    tiles = pkl.load(open(DATA_DIR+"/vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open(DATA_DIR+"/indMat{}.pkl".format(objid)))
    workers = pkl.load(open(DATA_DIR+"/worker{}.pkl".format(objid)))

    tile_area = np.array(indMat[-1])
    A_thres = np.percentile(tile_area,A_percentile)
    Qj_lst=[]
    #if DEBUG: print "Coming up with T' combinations to search through"
    #Tprime_lst, Tprime_idx_lst = Tprimefunc(objid,tiles,indMat,fixedtopk=3, topk = 40,NTprimes=NTprimes)
    Tstar_lst = []
    Tstar_idx_lst =[]
    likelihood_lst=[]

    if DEBUG: print "Initializing tiles "
    Tstar,Tidx=initT(tiles,indMat)
    Tstar_lst.append([Tstar])
    Tstar_idx_lst.append(Tidx)

    for i in tqdm(range(Niter)):
        if DEBUG: print "Iteration #", i
        plk=0
        Tidx_lst = []

        if DEBUG: print "E-step : Estimate Qj parameters"
        Qjhat = estimate_Qj(Tstar,tiles,indMat,workers,Qjfunc,A_percentile,DEBUG=DEBUG)
        Qn1,Qn2,Qp1,Qp2 = zip(*Qjhat)

        if DEBUG: print "ML construction of Tstar"
        dPrime = 0

        exclude_idx = set(Tstar_idx_lst[0])
        good_dPrime_tcount = len(exclude_idx)
        current_shell_tkidxs= Tidx
        past_shell_tkidxs= Tidx
        while (good_dPrime_tcount!=0 or len(current_shell_tkidxs)!=0):

            print "Excluding",exclude_idx
            current_shell_tkidxs = find_all_tk_in_shell(tiles,past_shell_tkidxs,list(exclude_idx))

            if DEBUG: 
                print "d'={0}; good_dPrime_tcount={1}".format(dPrime,good_dPrime_tcount)
                print "Number of tks in shell: ",len(current_shell_tkidxs)
            good_dPrime_tcount=0

            #print "First occurence of tk satisfying criterion"
            Tstar_lst.append([Tstar_lst[0][0]])

            for k in current_shell_tkidxs:
                pInT = 0
                pNotInT = 0
                tk = tiles[k]
                # Compute pInT and pNotInT
                for j in range(len(workers)):
                    ljk = indMat[j][k] #NOTE k doesn't correspond to k in tiles but in current_shell_tks so this is not good
                    wid=workers[j] 
                    qp1 = Qp1[j]
                    qp2 = Qp2[j]
                    qn1 = Qn1[j]
                    qn2 = Qn2[j]
                    if tk.area>A_thres:
                        if ljk ==1:
                            pInT+=np.log(qp1)
                            pNotInT+=np.log(1-qn1)
                        else:
                            pInT+=np.log(1-qp1)
                            pNotInT+=np.log(qn1)
                    else:
                        if ljk ==1:
                            pInT+=np.log(qp2)
                            pNotInT+=np.log(1-qn2)
                        else:
                            pInT+=np.log(1-qp2)
                            pNotInT+=np.log(qn2)
                # Check if tk satisfy constraint
                
                if pInT>=pNotInT:
                    
                    # if satisfy criterion, then add to Tstar
                    plk+=pInT+pNotInT
                    good_dPrime_tcount+=1
                    if DEBUG: print "Adding tk",k
                    try:
                        Tstar_lst[i]=[Tstar_lst[i][0].union(tk)]
                        Tidx_lst.append(k)
                    except(shapely.errors.TopologicalError):
                        try:
                            Tstar_lst[i]=[Tstar_lst[i][0].buffer(0).union(tk.buffer(-1e-10))]
                            Tidx_lst.append(k)
                        except(shapely.errors.TopologicalError):
                            try:
                                Tstar_lst[i]=[Tstar_lst[i][0].buffer(-1e-10).union(tk)]
                                Tidx_lst.append(k)
                            except(shapely.errors.TopologicalError):
                                try:
                                    Tstar_lst[i]=[Tstar_lst[i][0].buffer(-1e-10).union(tk.buffer(-1e-10))]
                                    Tidx_lst.append(k)
                                except(shapely.errors.TopologicalError):
                                    try:
                                        Tstar_lst[i]=[Tstar_lst[i][0].union(tk.buffer(1e-10))]
                                        Tidx_lst.append(k)
                                    except(shapely.errors.TopologicalError):
                                        try:
                                            Tstar_lst[i]=[Tstar_lst[i][0].buffer(1e-10).union(tk)]
                                            Tidx_lst.append(k)
                                        except(shapely.errors.TopologicalError):
                                            try:
                                                Tstar_lst[i]=[Tstar_lst[i][0].buffer(1e-10).union(tk.buffer(1e-10))]
                                                Tidx_lst.append(k)
                                            except(shapely.errors.TopologicalError):
                                                print "Shapely Topological Error: unable to add tk, Tstar unchanged; at k=",k
                                                pkl.dump(Tstar_lst[i][0],open("problematic_Tstar_{0}.pkl".format(k),'w'))
                                                pkl.dump(tk,open("problematic_tk_{0}.pkl".format(k),'w'))
                                                pass

            ############################################################################################################
            if PLOT:
                plt.figure()
                for c in current_shell_tkidxs:plot_coords(tiles[c],color="red",fill_color="red") #current front
                for c in past_shell_tkidxs:plot_coords(tiles[c],color="cyan",linewidth=5,linestyle='--') #past front
                for c in exclude_idx:plot_coords(tiles[c],color="gray",fill_color="gray")#excluded coord
                plot_coords(Tstar_lst[i][0],linestyle="--",linewidth=2,color="blue")#current Tstar
                for c in Tidx_lst:plot_coords(tiles[c],linewidth=2,color="green",fill_color="green")#new Tstar
                plt.ylim(40,100)


            
            #Updates
            Tstar = Tstar_lst[i][0].buffer(0)
            dPrime+=1
            past_shell_tkidxs= current_shell_tkidxs
            exclude_idx= exclude_idx.union(current_shell_tkidxs)
        #Storage
        Qj_lst.append(Qjhat)
        Tstar_idx_lst.append(Tidx_lst)
        likelihood_lst.append(plk)

    return Tstar_idx_lst , likelihood_lst, Qj_lst,Tstar_lst