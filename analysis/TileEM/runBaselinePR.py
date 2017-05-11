# Compute PR from all baselines + TileEM
import pickle as pkl 
from TileEM_plot_toolbox import *
from qualityBaseline import *
from glob import glob
from collections import OrderedDict
from bad_vtiles import * 
from calc_Tstar import *
import time
df = pd.read_csv("../computed_my_COCO_BBvals.csv",index_col=0)
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
#mode="aggregate_sample_table"
#mode="recompute_sample_batch_table"
mode="concatenate_sample_table"
base_dir= "uniqueTiles"
discard_obj_lst = [35,40,41]
def majority_vote(objid,heuristic="50%"):
    #Compute PR for majority voted region
    tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
    indMat = pkl.load(open("indMat{}.pkl".format(objid)))
    workers = pkl.load(open("worker{}.pkl".format(objid)))

    area = indMat[-1]
    votes = indMat[:-1].sum(axis=0)
    if heuristic=="50%":
        tidx=np.where(votes>np.shape(indMat[:-1])[0]/2.)[0]
    elif heuristic=="topk":
        topk=10
        tidx = np.argsort(votes)[::-1][:topk]
    elif heuristic=="topPercentile":
        percentile=95
        tidx=np.where(votes>np.percentile(votes,percentile))[0]
    P,R = compute_PR(objid,tidx,tiles)
    if len(tidx)==0:
        P=0
        R=0
    return P,R
def precision_from_list(test_list, base_poly):
    int_area = 0.0
    test_poly_area = 0.0
    for test_poly in test_list:
        int_area += intersection_area(test_poly, base_poly)
        test_poly_area += test_poly.area
    return (int_area / test_poly_area) if (test_poly_area != 0) else 0

def recall_from_list(test_list, base_poly):
    int_area = 0.0
    for test_poly in test_list:
        int_area += intersection_area(test_poly, base_poly)
    return (int_area / base_poly.area) if (base_poly.area != 0) else 0


if mode =="recompute_sample_batch_table":
    start = time.time()
    # Compute and save a  PR_tbl_all.csv into each sample_batch folder
    PR_non_adjacency = pd.read_csv("COMPILED_PR_nadj.csv")
    PR_adjacency = pd.read_csv("COMPILED_PR_adj.csv")
    start = time.time()
    # Compute and save a  PR_tbl_all.csv into each sample_batch folder
    for Nworker in sampleN_lst:
        for batch_num in range(worker_Nbatches[Nworker]):
            dir_name = "{0}worker_rand{1}".format(Nworker,batch_num)
            print "Working on :", dir_name
            os.chdir(base_dir+"/"+dir_name)
            # Creating 2 empty precision recall table containing P,R for each metric used 
            cols = [u'Num Points',u'Area Ratio',u'Jaccard [Self]', u'Precision [Self]', u'Recall [Self]']

            PR_tbl = pd.DataFrame()#index=object_lst)#,columns=cols)
            PR_tbl["object_id"]=object_lst
            # Summarization-Based Method
            df.ix[df["Precision [Self]"]>1,"Precision [Self]"]=1
            df.ix[df["Recall [Self]"]>1,"Recall [Self]"]=1
            df = df[~(df["worker_id"].isin([1,2,3]))] #Filter our ground truth workers
            for attr in cols:
                tbl=[]
                for objid in object_lst:
                    workers=pkl.load(open("worker{}.pkl".format(objid)))
                    filtered_df = df[(df["worker_id"].isin(workers))&(df["object_id"]==objid)] #only look at summarization scores of sampled workers
                    best_worker_BB = filtered_df[filtered_df[attr]==filtered_df[attr].max()]
                    if objid not in [35,41,40]:
                        tbl.append([objid,best_worker_BB["Precision [Self]"].values[0],best_worker_BB["Recall [Self]"].values[0]])
                    else:
                        tbl.append([objid,-1,-1])
                tmp_PR_tbl = pd.DataFrame(tbl,columns=["object_id","Precision","Recall"])
                PR_tbl["P [{}]".format(attr)]=tmp_PR_tbl["Precision"]
                PR_tbl["R [{}]".format(attr)]=tmp_PR_tbl["Recall"]
            # Vision based methods
            visionPR = pd.read_csv("../../Vision-stuff/VisionGTComparisons/{}/best_Area_Ratio_BBs/box/prec50/PR.csv".format(dir_name))
            PR_tbl["P [Vision Best Area Ratio 50%]"] = visionPR["precision"] 
            PR_tbl["R [Vision Best Area Ratio 50%]"] = visionPR["recall"]

            visionPR = pd.read_csv("../../Vision-stuff/VisionGTComparisons/{}/best_Num_Points_BBs/box/prec50/PR.csv".format(dir_name))
            PR_tbl["P [Vision Best Num Points 50%]"] = visionPR["precision"]
            PR_tbl["R [Vision Best Num Points 50%]"] = visionPR["recall"]

            for threshold in [10,50,90]:
                visionPR = pd.read_csv("../../../PR{}.csv".format(threshold))
                PR_tbl["P [Vision GT {}%]".format(threshold)] = visionPR["precision"]
                PR_tbl["R [Vision GT {}%]".format(threshold)] = visionPR["recall"]
            #Tile-based methods
            tbl=[]
            col_lst = []
            tmp_tbl=[]
            for i,fname in enumerate(glob.glob("obj*")):
                objid=int(fname[3:])
                BBG =get_gt(objid)
                tmp_tbl=[objid]
                if i==0: col_lst = ["object_id"]
                tiles = pkl.load(open("vtiles{}.pkl".format(objid)))
                for thresh in [-40,-20,0,20,40]:
                    try:
                        #print "Working on obj{0}/thresh{1}/".format(objid,thresh)
                        #TileEMP,TileEMR = compute_PR(objid,np.array(Tstar_lst),tiles)
                        # Tile EM with adjacency
                        selected_PR = PR_adjacency[(PR_adjacency["num_workers"]==Nworker)&(PR_adjacency["sample_num"]==batch_num)&\
                                         (PR_adjacency["objid"]==objid)&(PR_adjacency["iter_num"]==5)&\
                                         (PR_adjacency["thresh"]==thresh)]
                        if len(selected_PR)==1:
                            TileEMP = float(selected_PR["precision"])
                            TileEMR = float(selected_PR["recall"])
                        else:
                            TileEMP=-1
                            TileEMR=-1
                        # Tile EM without adjacency
                        selected_PR = PR_non_adjacency[(PR_non_adjacency["num_workers"]==Nworker)&(PR_non_adjacency["sample_num"]==batch_num)&\
                                         (PR_non_adjacency["objid"]==objid)&(PR_non_adjacency["iter_num"]==5)&\
                                         (PR_non_adjacency["thresh"]==thresh)]
                        if len(selected_PR)==1:
                            TileEMP_NA = float(selected_PR["precision"])
                            TileEMR_NA = float(selected_PR["recall"])
                        else:
                            TileEMP=-1
                            TileEMR=-1
                        tmp_tbl.extend([TileEMP,TileEMR,TileEMP_NA,TileEMR_NA])
                        if i==0: 
                            col_lst.extend(["P [TileEM thres={}]".format(thresh),"R [TileEM thres={}]".format(thresh),"P [TileEM NA thres={}]".format(thresh),"R [TileEM NA thres={}]".format(thresh)])
                    except(IOError):
                        print "No file exist: obj{0}/thresh{1}/iter_5/tid_list.pkl".format(objid,thresh)
                        pass
                # Majority Vote
                if False:
		    PMVT,RMVT = majority_vote(objid,heuristic="50%")
                    PMVTtopk,RMVTtopk = majority_vote(objid,heuristic="topk")
                    PMVTtopP,RMVTtopP = majority_vote(objid,heuristic="topPercentile")
                    tmp_tbl.extend([PMVT,RMVT,PMVTtopk,RMVTtopk,PMVTtopP,RMVTtopP])
                    if i==0: col_lst.extend(["P [MVT]","R [MVT]","P [MVTtop10]","R [MVTtop10]","P [MVTtop95%]","R [MVTtop95%]"])
                tbl.append(tmp_tbl) #[objid,TileEMP,TileEMR,PMVT,RMVT,PMVTtopk,RMVTtopk,PMVTtopP,RMVTtopP]
            Tile_df = pd.DataFrame(tbl,columns=col_lst)
	    #print col_lst
            #Tile_df.to_csv("Tile_PR.csv")
	    PR_tbl_all = PR_tbl.merge(Tile_df,on="object_id")
	    #Save to file in that folder 
            PR_tbl_all.to_csv("PR_tbl_all.csv")
            os.chdir("../../") 
elif mode =="aggregate_sample_table" :
    for Nworker in sampleN_lst:
	print "Working on worker = ",Nworker
        batch_all_data=[]
        for batch_num in range(worker_Nbatches[Nworker]):
            dir_name = "{0}/{1}worker_rand{2}".format(base_dir,Nworker,batch_num)
            batch_i_data=pd.read_csv(dir_name+"/PR_tbl_all.csv",index_col=0)
            #Drop the object rows where the objects have bad vtiles to begin with
            bad_vtile_objs = list(problematic[(problematic["Nworker"]==Nworker)&(problematic["batch_num"]==batch_num)].objid)
            bad_vtile_objs.extend(discard_obj_lst) # Discard ambiguous objects
            batch_i_data=batch_i_data.drop(bad_vtile_objs,errors='ignore')

            batch_all_data.append(batch_i_data)
        try:
            batch_all_data = pd.concat(batch_all_data, axis=0, join='outer', join_axes=None, ignore_index=False,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
            batch_all_data.to_csv("concat_sample{}_PR.csv".format(Nworker))
        except(ZeroDivisionError):
            print "No data for worker=",Nworker
    sampleN_lst=sorted(worker_Nbatches.keys())

    for Nworker in sampleN_lst:
        print "Working on worker = ",Nworker
        batch_all_data=[]
        for batch_num in range(worker_Nbatches[Nworker]):
            dir_name = "sample/{0}worker_rand{1}".format(Nworker,batch_num)
            batch_i_data=pd.read_csv(dir_name+"/PR_tbl_all.csv",index_col=0)
            #Drop the object rows where the objects have bad vtiles to begin with 
            bad_vtile_objs = list(problematic[(problematic["Nworker"]==Nworker)&(problematic["batch_num"]==batch_num)].objid)
	    bad_vtile_objs.extend(discard_obj_lst) # Discard ambiguous objects 
            batch_i_data=batch_i_data.drop(bad_vtile_objs,errors='ignore')
            #batch_i_data[~batch_i_data["object_id"].isin(bad_vtile_objs)]
            # First set the object_id column the index, then reindex based on this index to correspond to object lst
            # This fills in a row of NaN for the missing data objects, so it normalizes the shape of our data table 
            batch_i_data = batch_i_data.set_index('object_id').reindex(object_lst,fill_value =0)
             # Add a row of non-nan count for averaging 
            nan_rowidx= list(batch_i_data[batch_i_data["P [Num Points]"]==0].index)
            non_nan_count = np.ones_like(object_lst)
            #non_nan_count = np.ones(47)
	    for row in nan_rowidx: non_nan_count[row-1]=0
            batch_i_data["non_nan_count"]=non_nan_count
            if batch_num==0:
                batch_all_data=batch_i_data
            else:
                batch_all_data+=batch_i_data
        try:
            batch_all_data=batch_all_data[batch_all_data.keys()[:-1]].divide(batch_all_data["non_nan_count"],axis=0)
            batch_all_data[batch_all_data>1]=1
            batch_all_data[batch_all_data<0]=0
	    print "Created sample{}_PR.csv".format(Nworker)
            batch_all_data.to_csv("sample{}_PR.csv".format(Nworker))
        except(ZeroDivisionError):
            print "No data for worker=",Nworker 
elif mode=="concatenate_sample_table":
    print "Concatenating Tables" 
    batch_all_sample_data=[]
    for Nworker in sampleN_lst:
        print "Working on worker = ",Nworker
        #batch_all_data=[]
        for batch_num in range(worker_Nbatches[Nworker]):
            dir_name = "{0}/{1}workers_rand{2}".format(base_dir,Nworker,batch_num)
            batch_i_data=pd.read_csv(dir_name+"/Tile_PR.csv",index_col=0)
	    #print dir_name
	    #print batch_i_data.head()
            #Drop the object rows where the objects have bad vtiles to begin with
            bad_vtile_objs = list(problematic[(problematic["Nworker"]==Nworker)&(problematic["batch_num"]==batch_num)].objid)
            bad_vtile_objs.extend(discard_obj_lst) # Discard ambiguous objects
            batch_i_data=batch_i_data.drop(bad_vtile_objs,errors='ignore')
	    if len(batch_i_data)!=0:
		print "Added :",dir_name
		batch_i_data["Nworker"]=Nworker
		batch_i_data["batch_num"]=batch_num
                #batch_all_data.append(batch_i_data)
     	        batch_all_sample_data.append(batch_i_data)
            #try:
            #    batch_all_data = pd.concat(batch_all_data, axis=0, join='outer', join_axes=None, ignore_index=False,
            #  keys=None, levels=None, names=None, verify_integrity=False,
            #  copy=True)
            #    batch_all_data.to_csv("concat_sample{}_PR.csv".format(Nworker))
            #except(ZeroDivisionError):
            #    print "No data for worker=",Nworker
    batch_all_sample_data = pd.concat(batch_all_sample_data, axis=0, join='outer', join_axes=None, ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, copy=True)
    #ordered_keys = list(batch_all_sample_data.keys()[:1])+list(batch_all_sample_data.keys()[-2:])+list(batch_all_sample_data.keys()[1:-2]) 
    #print ordered_keys
    #batch_all_sample_data= batch_all_sample_data.reindex_axis(ordered_keys,axis=1)
    batch_all_sample_data.to_csv("Tile_PR_all.csv")
