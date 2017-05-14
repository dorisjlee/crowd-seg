# Run the PR calculation for PixelEM on various thresholds based on  the adjacency and the non-adjacency ML construction
# Creates a Pixel_PR.csv in each of the base_dir/dir_name directory 
import pandas as pd 
import pickle as pkl 
from qualityBaseline import * 
from calc_Tstar import * 
import os 
import time 
from PixelEM import * 
base_dir= "pixel_em"
start = time.time()
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
from TileEM_plot_toolbox import *
start = time.time()
PR_pixelEM = pd.read_csv("pixel_em/full_PRJ_table.csv")
PR_pixelEM_GT = pd.read_csv("pixel_em/GTfull_PRJ_table.csv")
#PR_pixelEM_GTLSA = pd.read_csv("pixel_em/GTLSAfull_PRJ_table.csv")
df = pd.read_csv("../computed_my_COCO_BBvals.csv",index_col=0)
tbl=[]
col_lst=[]
i=0
cols = [u'Num Points',u'Area Ratio',u'Jaccard [Self]', u'Precision [Self]', u'Recall [Self]']
for Nworker in sampleN_lst:
    for batch_num in range(worker_Nbatches[Nworker]):
        dir_name = "{0}workers_rand{1}".format(Nworker,batch_num)
        print "Working on :", dir_name
        #os.chdir(base_dir+"/"+dir_name)
        tmp_tbl=[]
        #for i,fname in enumerate(glob("obj*")):
        #    objid=int(fname[3:])
	object_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
	for objid in tqdm(object_lst):
            tmp_tbl=[objid,Nworker,batch_num]
            if i==0: col_lst = ["object_id","Nworker","batch_num"]
	    #Summarization Based metrics
	    workers = ast.literal_eval([line for line in open("pixel_em/"+dir_name+"/obj{}/worker_ids.json".format(objid))][0])
            filtered_df = df[(df["worker_id"].isin(workers))&(df["object_id"]==objid)] #only look at summarization scores of sampled workers
	    #if i==0: col_lst.extend(cols)
	    if len(filtered_df)!=0:
	        for attr in cols:
		    if i==0:
		        col_lst.extend(["P [{}]".format(attr),"R [{}]".format(attr),"J [{}]".format(attr)])
                    best_worker_BB = filtered_df[filtered_df[attr]==filtered_df[attr].max()]
                    best_worker_id = int(best_worker_BB["worker_id"].sample(1).values)
                    best_worker_mask = pkl.load(open("pixel_em/obj{}/mask{}.pkl".format(objid,best_worker_id)))
                    gt_mask = pkl.load(open("pixel_em/obj{}/gt.pkl".format(objid)))
                    p,r,j = get_precision_recall_jaccard(best_worker_mask, gt_mask)
                    tmp_tbl.extend([p,r,j])
	        # Pixel EM
                for thresh in [-4,-2,0,2,4]:#,10,-10]:
                    #try:
		    if True:
                        #print "Working on obj{0}/thresh{1}/".format(objid,thresh)
		        PR_pixelEMi =PR_pixelEM[(PR_pixelEM["num_workers"]==Nworker)&(PR_pixelEM["sample_num"]==batch_num)&\
					(PR_pixelEM["objid"]==objid)&(PR_pixelEM["thresh"]==thresh)]
		        if len(PR_pixelEMi)!=0:
      		            PixelEMP= float(PR_pixelEMi["EM_precision"]) 
		            PixelEMR= float(PR_pixelEMi["EM_recall"])
		   	    PixelEMJ=float(PR_pixelEMi["EM_jaccard"])
                            tmp_tbl.extend([PixelEMP,PixelEMR,PixelEMJ])
                            if i==0: 
                                 col_lst.extend(["P [PixelEM thres={}]".format(thresh),"R [PixelEM thres={}]".format(thresh),"J [PixelEM thres={}]".format(thresh)])
                    #except(IOError):
                    #    print "No file exist: obj{0}/thresh{1}/iter_5/tid_list.pkl".format(objid,thresh)
                    #    pass
                # Majority Vote
	        if len(PR_pixelEMi)!=0:
                    PMV = float(PR_pixelEMi["MV_precision"])
        	    RMV = float(PR_pixelEMi["MV_recall"])
		    JMV = float(PR_pixelEMi["MV_jaccard"])
                    tmp_tbl.extend([PMV,RMV,JMV])
                    if i==0: col_lst.extend(["P [MV]","R [MV]","J [MV]"])
		    #if len(tmp_tbl)==len(col_lst):
                tbl.append(tmp_tbl)
   	        i+=1
df = pd.DataFrame(tbl,columns=col_lst)
df.to_csv("Pixel_PR.csv")
#os.chdir("../../")
end = time.time()
print "Time Elapsed: ", end-start 
