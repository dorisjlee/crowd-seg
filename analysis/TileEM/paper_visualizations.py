# Visualization for paper and debugging 
from TileEM_plot_toolbox import *
from TileEM import * 
from qualityBaseline import *
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
Nsample_lst = worker_Nbatches.keys()
Tile_tbl = pd.read_csv("Tile_PR_all.csv",index_col=0)

Tile_tbl = Tile_tbl.rename(index=str,columns={'P [TileEM thres=-40]':'P [TileEM thres=-4]',\
                                               'P [TileEM thres=-20]':'P [TileEM thres=-2]',\
                                               'R [TileEM thres=-40]':'R [TileEM thres=-4]',\
                                              'R [TileEM thres=-20]':'R [TileEM thres=-2]',\
                                              'J [TileEM thres=-40]':'J [TileEM thres=-4]',\
                                              'J [TileEM thres=-20]':'J [TileEM thres=-2]',\
                                              'P [TileEM thres=40]':'P [TileEM thres=4]',\
                                               'P [TileEM thres=20]':'P [TileEM thres=2]',\
                                               'R [TileEM thres=40]':'R [TileEM thres=4]',\
                                              'R [TileEM thres=20]':'R [TileEM thres=2]',\
                                              'J [TileEM thres=40]':'J [TileEM thres=4]',\
                                              'J [TileEM thres=20]':'J [TileEM thres=2]'})
Pixel_tbl = pd.read_csv("updated_Pixel_PR.csv",index_col=0)
#PR_tbl = PR_tbl.rename(index=str,columns={'GT Tile-based Precision':'P [GT Tile-based]','GT Tile-based Recall':'R [GT Tile-based]'})
df_all = Pixel_tbl.merge(Tile_tbl)

def data_clean(df):
    df = df.rename(index=str,columns={'P [Jaccard [Self]]':'P [GT Jaccard]','R [Jaccard [Self]]':'R [GT Jaccard]',\
                             'P [Precision [Self]]':'P [GT Precision]','R [Precision [Self]]':'R [GT Precision]',\
                             'P [Recall [Self]]':'P [GT Recall]','R [Recall [Self]]':'R [GT Recall]',\
                            })
    return df
def selected_attr2col_lst(selected_attr_lst, attr_only=['P','R','J']):
    selected_col_lst =[]
    for attr in selected_attr_lst:
        if 'P' in attr_only: selected_col_lst.append("P [{}]".format(attr))
        if 'R' in attr_only: selected_col_lst.append("R [{}]".format(attr))
        if 'J' in attr_only: selected_col_lst.append("J [{}]".format(attr))
    return selected_col_lst
def plot_PR(Nsample,selected_attr_lst):
    df = pd.read_csv("sample{}_PR.csv".format(Nsample))
    df=data_clean(df)
    plt.figure()
    plt.title("{0} worker sample averaged over {1} batches".format(Nsample,worker_Nbatches[Nsample]))
    for attr in selected_attr_lst:
        plt.plot(df["R [{}]".format(attr)],df["P [{}]".format(attr)],'.',label=attr)
    plt.xlabel("Recall",fontsize=14)
    plt.ylabel("Precision",fontsize=14)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

def plot_sample_worker_PR(selected_col_lst,y_axis='Precision',init_fig=True,color='blue'):
    if init_fig: plt.figure()
    df_all = pd.DataFrame() #mean 
    df_all_std = pd.DataFrame() # std
    cols  = []
    y_err_lst = []
    y_val_lst =[]
    sample_lst = sorted(Nsample_lst)[:4]
    for Nsample in sample_lst:
        df = pd.read_csv("concat_sample{}_PR.csv".format(Nsample))
        df=data_clean(df)
        cols.append(Nsample)
        y_val_lst.append(list(df[selected_col_lst].mean()))
        y_err_lst.append(list(df[selected_col_lst].std()))

    y_val_lst = np.array(y_val_lst).T
    y_err_lst = np.array(y_err_lst).T
    for i in range(len(y_val_lst)):
        plt.plot(sample_lst,y_val_lst[i],label=selected_col_lst[i])
        #plt.errorbar(sample_lst,y_val_lst[i],label=selected_col_lst[i], yerr=y_err_lst[i],capsize=3)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.xlabel("N sample",fontsize=13)
    plt.ylabel(y_axis,fontsize=13)
    #plt.ylim(0,1)

def plot_sample_worker_PR_subplots():
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    y_axis="Precision"
    selected_col_lst =[]
    for attr in selected_attr_lst:
        if y_axis=="Precision":selected_col_lst.append("P [{}]".format(attr))
        if y_axis=="Recall":selected_col_lst.append("R [{}]".format(attr))
    df_all = pd.DataFrame() #mean 
    df_all_std = pd.DataFrame() # std
    cols  = []
    y_err_lst = []
    y_val_lst =[]
    sample_lst = sorted(Nsample_lst)[:4]
    for Nsample in sample_lst:
        df = pd.read_csv("concat_sample{}_PR.csv".format(Nsample))
        df=data_clean(df)
        cols.append(Nsample)
        y_val_lst.append(list(df[selected_col_lst].mean()))
        y_err_lst.append(list(df[selected_col_lst].std()))

    y_val_lst = np.array(y_val_lst).T
    y_err_lst = np.array(y_err_lst).T
    lines=[]
    for i in range(len(y_val_lst)):
        lines.append(ax1.plot(sample_lst,y_val_lst[i])[0])#,label=selected_col_lst[i])
        #plt.errorbar(sample_lst,y_val_lst[i],label=selected_col_lst[i], yerr=y_err_lst[i],capsize=3)
    ax1.legend(lines,selected_attr_lst, loc="lower left",fontsize=9)
    ax1.set_xlabel("N sample",fontsize=13)
    ax1.set_ylabel(y_axis,fontsize=13)
    #plt.ylim(0,1)
    #ax2 =plt.subplot(2, 1, 2)
    y_axis='Recall'

    selected_col_lst =[]
    for attr in selected_attr_lst:
        if y_axis=="Precision":selected_col_lst.append("P [{}]".format(attr))
        if y_axis=="Recall":selected_col_lst.append("R [{}]".format(attr))
    df_all = pd.DataFrame() #mean 
    df_all_std = pd.DataFrame() # std
    cols  = []
    y_err_lst = []
    y_val_lst =[]
    sample_lst = sorted(Nsample_lst)[:4]
    for Nsample in sample_lst:
        df = pd.read_csv("concat_sample{}_PR.csv".format(Nsample))
        df=data_clean(df)
        cols.append(Nsample)
        y_val_lst.append(list(df[selected_col_lst].mean()))
        y_err_lst.append(list(df[selected_col_lst].std()))

    y_val_lst = np.array(y_val_lst).T
    y_err_lst = np.array(y_err_lst).T
    for i in range(len(y_val_lst)):
        ax2.plot(sample_lst,y_val_lst[i])#,label=selected_attr_lst[i])
        #plt.errorbar(sample_lst,y_val_lst[i],label=selected_col_lst[i], yerr=y_err_lst[i],capsize=3)
    ax2.set_xlabel("N sample",fontsize=13)
    ax2.set_ylabel(y_axis,fontsize=13)
    f.savefig('../../docs/overleaf_paper/plots/PRsample.pdf')
def plot_PR_vary_sample_size(algorithm):
    plt.figure()
    plt.title("PR across different worker samples [{}]".format(algorithm),fontsize=14)

    for Nsample in Nsample_lst:
        df = pd.read_csv("sample{}_PR.csv".format(Nsample))
        df=data_clean(df)
        plt.plot(df["R [{}]".format(algorithm)],df["P [{}]".format(algorithm)],'.',label="N={}".format(Nsample))

    plt.xlabel("Recall",fontsize=13)
    plt.ylabel("Precision",fontsize=13)
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
def plot_vary_threshold(Nworker):
    plt.figure()
    tbl= pd.read_csv("concat_sample{}_PR.csv".format(Nworker),index_col=0)
    thres_lst = [-40,-20,0,20,40]
    plt.plot(thres_lst,list(tbl.filter(regex="TileEM").filter(regex="P").mean()),label="Precision")
    plt.plot(thres_lst,list(tbl.filter(regex="TileEM").filter(regex="R").mean()),label="Recall")
    plt.title("N worker={}".format(Nworker),fontsize=14)
    plt.xlabel("Threshold",fontsize=13)
    plt.legend()

# viz for debugging 

def plot_masks(batch,objid,thresh,algo,include=['pNInT',"mega","gtResult"],returnMatrix=False):    
    if thresh ==0:
        thresh=float(thresh)
    if returnMatrix : 
	returnMatLst=[]
    if 'mega' in include:
        mega_mask = pkl.load(open("pixel_em/{}/obj{}/mega_mask.pkl".format(batch,objid)))
        plt.figure()
        plt.imshow(mega_mask)
        plt.title("Mega mask")
        plt.colorbar()
	if returnMatrix: returnMatLst.append(mega_mask)
    if ('gtResult' in include) or ('ResultOnly' in include):
	result = pkl.load(open("pixel_em/{}/obj{}/{}_gt_est_ground_truth_mask_thresh{}.pkl".format(batch,objid,algo,thresh)))
	if returnMatrix: returnMatLst.append(result)
        plt.figure()
	if 'ResultOnly' in include:
	    plt.title("EM Result")
        else:
	    gt = pkl.load(open("pixel_em/obj{}/gt.pkl".format(objid)))
	    plt.title("GT [cyan] & EM Result[brighter yellow]")
	    plt.imshow(gt,alpha=0.4)
	    plot_coords(ground_truth_T(objid,reverse_xy=True),'cyan')
	    if returnMatrix: returnMatLst.append(gt)
        plt.imshow(result)
        plt.colorbar()
    if 'pNInT' in include:
        pInT =  pkl.load(open("pixel_em/{}/obj{}/{}_p_in_mask_ground_truth_thresh{}.pkl".format(batch,objid,algo,thresh)))
        pNotInT =  pkl.load(open("pixel_em/{}/obj{}/{}_p_not_in_ground_truth_thresh{}.pkl".format(batch,objid,algo,thresh)))
        plt.figure()
        plt.title("pInT")
        plt.imshow(pInT)
        plt.colorbar()
	
        plt.figure()
        plt.title("pNotInT")
        plt.imshow(pNotInT)
        plt.colorbar()
	if returnMatrix: 
	    returnMatLst.append(pInT)
	    returnMatLst.append(pNotInT)
    if returnMatrix: return returnMatLst

def plot_comparison(df,x_attr,y1_attr,y2_attr):
    if x_attr=="index":
        a = plt.plot(df.index,df[y1_attr],'.',label=y1_attr)
        a = plt.plot(df.index,df[y2_attr],'x',label=y2_attr )
    else:
        a = plt.plot(df[x_attr],df[y1_attr],'.',label=y1_attr)
        a = plt.plot(df[x_attr],df[y2_attr],'x',label=y2_attr )
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    plt.xlabel(x_attr)

def compare_PRJ_fixed_attr(df,attr,y1,y2,plot_metric='PRJ'):
    attr_lst = list(set(df[attr]))
    if 'P' in plot_metric:
        plt.figure()
        plt.title('Precision')
        plot_comparison(df,attr,"P [{}]".format(y1),"P [{}]".format(y2))
        plt.xticks(attr_lst,rotation='vertical')
    if 'R' in plot_metric: 
        plt.figure()
        plt.title('Recall')
        plot_comparison(df,attr,"R [{}]".format(y1),"R [{}]".format(y2))
        plt.xticks(attr_lst,rotation='vertical')
    if 'J' in plot_metric:
        plt.figure()
        plt.title('Jaccard')
        plot_comparison(df,attr,"J [{}]".format(y1),"J [{}]".format(y2))
        plt.xticks(attr_lst,rotation='vertical')
def compare_PRJ_fixed_sample_object(df,y1,y2):
    plt.figure()
    plot_comparison(df,"index","P [{}]".format(y1),"P [{}]".format(y2))
    plt.figure()
    plot_comparison(df,"index","R [{}]".format(y1),"R [{}]".format(y2))
    plt.figure()
    plot_comparison(df,"index","J [{}]".format(y1),"J [{}]".format(y2))

def plot_PRcurve(df,objid,num_worker,sample_num=0):
    objdf = df[(df["num_workers"]==num_worker)&(df["sample_num"]==sample_num)&(df["objid"]==objid)]
    plt.figure()
    for algo in ['basic','GT','isoGT','GTLSA','isoGTLSA','AW','isoAW']:
        x= objdf["P [{}]".format(algo)]
        y = objdf["R [{}]".format(algo)]
        if len(x)<=0:
            return
        sortedx, sortedy = zip(*sorted(zip(x, y)))
        plt.plot(sortedx,sortedy,'.-',label=algo)
    plt.xlabel("Precision",fontsize=13)
    plt.ylabel("Recall",fontsize=13)
    plt.legend(loc="bottom left")
    plt.title("{}worker_rand{} [obj {};N={}]".format(num_worker,sample_num,objid,len(objdf)))

