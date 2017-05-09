# Run the PR calculation for PixelEM on various thresholds based on  the adjacency and the non-adjacency ML construction
# Creates a Pixel_PR.csv in each of the base_dir/dir_name directory 
import pandas as pd 
import pickle as pkl 
from qualityBaseline import * 
from calc_Tstar import * 
import os 
import time 
base_dir= "pixel_em"
start = time.time()
worker_Nbatches={5:10,10:8,15:6,20:4,25:2,30:1}
sampleN_lst=worker_Nbatches.keys()
from TileEM_plot_toolbox import *
start = time.time()
PR_pixelEM = pd.read_csv("pixel_em/full_PRJ_table.csv")
tbl=[]
col_lst=[]
i=0

for Nworker in sampleN_lst:
    for batch_num in range(worker_Nbatches[Nworker]):
        dir_name = "{0}worker_rand{1}".format(Nworker,batch_num)
        print "Working on :", dir_name
        #os.chdir(base_dir+"/"+dir_name)
        tmp_tbl=[]
        #for i,fname in enumerate(glob("obj*")):
        #    objid=int(fname[3:])
	for objid in range(1,49):
            tmp_tbl=[objid]
            if i==0: col_lst = ["object_id"]
            for thresh in [-4,-2,0,2,4,10,-10]:
                try:
                    #print "Working on obj{0}/thresh{1}/".format(objid,thresh)
		    # Pixel EM 
		    PR_pixelEMi =PR_pixelEM[(PR_pixelEM["num_workers"]==Nworker)&(PR_pixelEM["sample_num"]==batch_num)&\
					(PR_pixelEM["objid"]==objid)&(PR_pixelEM["thresh"]==thresh)]
		    if len(PR_pixelEMi)!=0:
      		        PixelEMP= float(PR_pixelEMi["EM_precision"]) 
		        PixelEMR= float(PR_pixelEMi["EM_recall"])
			PixelEMJ=float(PR_pixelEMi["EM_jaccard"])
                        tmp_tbl.extend([PixelEMP,PixelEMR,PixelEMJ])
                        if i==0: 
                             col_lst.extend(["P [PixelEM thres={}]".format(thresh),"R [PixelEM thres={}]".format(thresh),"J [PixelEM thres={}]".format(thresh)])
                except(IOError):
                    print "No file exist: obj{0}/thresh{1}/iter_5/tid_list.pkl".format(objid,thresh)
                    pass
            # Majority Vote
	    if len(PR_pixelEMi)!=0:
                PMV = float(PR_pixelEMi["MV_precision"])
        	RMV = float(PR_pixelEMi["MV_recall"])
		JMV = float(PR_pixelEMi["MV_jaccard"])
                tmp_tbl.extend([PMV,RMV,JMV])
                if i==0: col_lst.extend(["P [MV]","R [MV]","J [MV]"])
		if len(tmp_tbl)==len(col_lst):
                    tbl.append(tmp_tbl)
    		    i+=1
df = pd.DataFrame(tbl,columns=col_lst)
df.to_csv("Pixel_PR.csv")
#os.chdir("../../")
end = time.time()
print "Time Elapsed: ", end-start 
