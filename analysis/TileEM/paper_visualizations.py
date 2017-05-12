from TileEM_plot_toolbox import *
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
Pixel_tbl = pd.read_csv("Pixel_PR.csv",index_col=0)
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

def plot_sample_worker_PR(selected_col_lst,y_axis='Precision'):
    plt.figure()
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
